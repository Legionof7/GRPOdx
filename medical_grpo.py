# -*- coding: utf-8 -*-
"""Doctor–Patient GRPO (Multi-Turn) with a base reward for any final diagnosis + 5 completions per scenario.
   Modified to simulate patient responses via OpenAI API.
"""

########################################
# 0. Imports & Setup
########################################

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
from datasets import Dataset
import torch
import pandas as pd

import random
import re
import os
from contextlib import nullcontext
import datetime

from transformers import GenerationConfig
from trl import GRPOConfig
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from trl import maybe_apply_chat_template
from trl.trainer.grpo_trainer import pad

from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothGRPOTrainer

# NEW: Import OpenAI
from openai import OpenAI

# Initialize API key as None, will be set later
openai_api_key = None
client = None  # Will be initialized when API key is set

def initialize_openai_client(api_key):
    """Initialize the OpenAI client with the given API key"""
    global openai_api_key, client
    openai_api_key = api_key
    if api_key:
        client = OpenAI(api_key=api_key)

print("Imports complete.")

########################################
# 1. Load Base Model & LoRA
########################################

save_path = "/content/drive/MyDrive/UnslothGRPO/doctorExample"

max_seq_length = 2048
lora_rank = 32

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading base model ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,   # 4-bit quant
    fast_inference = True, # vLLM backend
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,
)

print("Attaching LoRA ...")
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("Model + LoRA ready.")

########################################
# 2. Doctor–Patient Scenario
########################################

COMMON_DISEASES = [
    "Influenza",
    "Common cold",
    "Strep throat",
    "COVID-19",
    "Allergic rhinitis",
    "Migraine",
    "Mononucleosis",
]

SYSTEM_PROMPT = """
You are a Doctor diagnosing a patient. 
Ask the patient questions, one at a time, to get more information and to rule things out. You can ask up to 4 questions. Ask as many questions as it takes to be sure on the diagnosis. 

When you know what the exact condition name is, provide a final line like:
Final diagnosis: XYZ
"""

MAX_TURNS = 5

class DoctorGame:
    """
    Multi-turn scenario:
      - A hidden disease (condition) is generated by the OpenAI API if an API key is provided,
        otherwise chosen randomly from COMMON_DISEASES.
      - The Doctor must produce <reason> blocks + visible text.
      - The episode ends if "Final diagnosis: X" is produced or MAX_TURNS is exceeded.
      - Partial credit and a base reward are provided if any final diagnosis is given.
    """

    def __init__(self, openai_api_key: str = None):
        global client
        if openai_api_key:
            try:
                # Initialize client with the provided API key
                local_client = OpenAI(api_key=openai_api_key)
                prompt = ("Generate a medical condition (for example: Influenza, COVID-19, Migraine, etc.) "
                          "and provide only the name of the condition.")
                response = local_client.chat.completions.create(model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=10)
                self.hidden_disease = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API error in generating condition: {e}")
                self.hidden_disease = random.choice(COMMON_DISEASES)
        else:
            self.hidden_disease = random.choice(COMMON_DISEASES)
        self.turn_count = 0
        self.done = False

    def is_done(self):
        return self.done or (self.turn_count >= MAX_TURNS)

    def parse_final_diagnosis(self, text: str) -> str:
        """If there's 'Final diagnosis: X', return X. Else ''."""
        match = re.search(r"Final\s*diagnosis:\s*(.*)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def compute_reward(self, final_guess: str, conversation_history: list = None) -> float:
        """
        Uses GPT-4o-mini to evaluate the entire conversation and calculate a reward based on:
        1. Correctness of the diagnosis
        2. Quality and relevance of the questions asked
        
        Falls back to basic reward calculation if OpenAI API is unavailable.
        """
        if not final_guess:
            return 0.0
            
        global client
        if client is None or not conversation_history:
            # Fallback to basic reward calculation if API unavailable or no conversation history
            return self._basic_reward_calculation(final_guess)
            
        try:
            # Format the conversation history for the model
            formatted_conversation = "\n".join(conversation_history)
            
            prompt = f"""Evaluate this doctor-patient conversation as a medical expert:

Hidden condition: {self.hidden_disease}
Doctor's final diagnosis: {final_guess}

Conversation:
{formatted_conversation}

Provide a score from 0.0 to 1.0 based on:
1. Diagnosis accuracy (70% weight): How correct was the final diagnosis?
   -Heavily penalize incorrect diagnoses
2. Question quality (30% weight): Were the doctor's questions relevant and logical?
   - Poor questions should receive low scores
   - Good, targeted questions should receive high scores
   - Low quantity of questions should receive low scores if more questions would have been more helpful

YOUR RESPONSE MUST BE ONLY A SINGLE NUMBER BETWEEN 0 AND 1."""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical evaluator that provides numerical scores only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            # Extract just the number from the response
            text = response.choices[0].message.content.strip()
            import re
            match = re.search(r"([0-9]*\.?[0-9]+)", text)
            
            if match:
                score = float(match.group(0))
                return max(0.0, min(1.0, score))  # Ensure score is in [0,1]
            
            # If we couldn't find a number, fall back to basic calculation
            print("Failed to parse score from GPT-4o-mini response, using fallback calculation.")
            return self._basic_reward_calculation(final_guess)
            
        except Exception as e:
            print(f"Error in GPT-4o-mini reward calculation: {e}")
            return self._basic_reward_calculation(final_guess)
            
    def _basic_reward_calculation(self, final_guess: str) -> float:
        """Fallback basic reward calculation method."""
        base = 0.2
        guess_lower = final_guess.lower()
        disease_lower = self.hidden_disease.lower()

        if guess_lower == disease_lower:
            return 1.0
        if guess_lower in disease_lower or disease_lower in guess_lower:
            return 0.8
        return base

########################################
# 3. Multi-Turn Generation
########################################

class DoctorGRPOTrainer(UnslothGRPOTrainer):
    """
    Similar to Tic-Tac-Toe's CustomGRPOTrainer, but for the Doctor scenario.
    We override multi_turn_generation to include patient simulation via the OpenAI API.
    """

    def __init__(self, *args, game_object=None, **kwargs):
        self.game_object_factory = game_object
        super().__init__(*args, **kwargs)

    def simulate_patient_response(self, conversation_history: str, hidden_disease: str) -> str:
        """
        Uses the OpenAI API to generate a patient response. The prompt instructs the model
        to simulate a patient who has the hidden condition but does not reveal it.
        """
        try:
            global client
            if client is None and hasattr(self.args, 'openai_api_key') and self.args.openai_api_key:
                client = OpenAI(api_key=self.args.openai_api_key)
            
            if client is None:
                raise ValueError("OpenAI client not initialized. Please provide a valid API key.")
                
            prompt = (
                f"You are a patient who has the following condition: {hidden_disease}. "
                "Answer the doctor's questions by describing your symptoms and feelings in a realistic manner "
                "without explicitly mentioning your condition. The conversation so far is:\n"
                f"{conversation_history}\n"
                "Now, provide your next response as a message starting with 'Patient:'"
            )
            response = client.chat.completions.create(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a patient simulating your condition."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150)
            patient_text = response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error in simulating patient response: {e}")
            # Fallback to a default patient message if API call fails
            patient_text = "Patient: I still have the same symptoms, please refine your diagnosis."
        return patient_text + "\n"

    def multi_turn_generation(self, prompt, model, tokenizer, generation_config, max_new_tokens=50, game_object=None):
        print("============ Starting a new Doctor–Patient episode ============")
        
        game = self.game_object_factory() if self.game_object_factory else None
        if not game:
            raise ValueError("No game_object_factory provided")
            
        print(f"Hidden disease: {game.hidden_disease}")
        print("---------------------------------------------------------")

        # Generate initial symptoms based on the disease
        initial_symptoms = generate_initial_symptom(game.hidden_disease)
        print(f"Initial patient symptoms: {initial_symptoms}")
        
        # Replace the placeholder with actual symptoms
        if "[PATIENT_SYMPTOMS]" in prompt:
            full_text = prompt.replace("[PATIENT_SYMPTOMS]", initial_symptoms)
        else:
            full_text = prompt
            
        total_reward = 0.0
        completion_ids = []
        conversation_history = []
        
        # Add initial symptoms to conversation history
        conversation_history.append(f"Patient initial query: {initial_symptoms}")

        while not game.is_done():
            outputs = self.llm.generate(
                [full_text],
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            new_text = outputs[0].outputs[0].text
            new_token_ids = outputs[0].outputs[0].token_ids

            full_text += new_text
            completion_ids.extend(new_token_ids)
            game.turn_count += 1
            
            # Add doctor's response to conversation history
            conversation_history.append(f"Doctor (Turn {game.turn_count}): {new_text}")

            # Check for final diagnosis in the doctor's output
            diag_match = re.search(r"Final\s*diagnosis:\s*(.*)", new_text, re.IGNORECASE)
            if diag_match:
                final_guess = diag_match.group(1).strip()
                game.done = True
                # Pass the conversation history for the reward model to evaluate
                diag_reward = game.compute_reward(final_guess, conversation_history)
                total_reward += diag_reward
                print(f"Doctor final guess: {final_guess} => reward={diag_reward:.3f}")
                break

            if game.turn_count >= MAX_TURNS:
                print("No final diagnosis after max turns => 0 reward")
                break

            # Instead of a hardcoded patient message, simulate the patient's response via OpenAI API.
            patient_text = self.simulate_patient_response(full_text, game.hidden_disease)
            full_text += patient_text
            
            # Add patient's response to conversation history
            conversation_history.append(f"Patient (Turn {game.turn_count}): {patient_text}")

        # Print the full conversation at the end
        print("\n========== Complete Conversation ==========")
        print(f"Actual condition: {game.hidden_disease}")
        print("-----------------------------------------")
        for message in conversation_history:
            print(message)
            print("-----------------------------------------")
        print(f"Total reward: {total_reward:.3f}")
        print("===========================================\n")

        return completion_ids, total_reward

    def _prepare_inputs(self, inputs: dict):
        """
        Identical approach to your Tic-Tac-Toe code:
          - Gather all prompts.
          - Run multi_turn_generation on the main process.
          - Parse final rewards.
          - Compute advantage.
          - Return RL data.
        """
        if not hasattr(self, "generation_config"):
            if self.args.use_vllm:
                self.generation_config = self.sampling_params
            else:
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=self.args.temperature,
                    pad_token_id=self.processing_class.pad_token_id,
                )
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] 
            for example in inputs
        ]

        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        )
        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs["attention_mask"].to(device)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)

            if self.accelerator.is_main_process:
                completion_ids_list = []
                game_rewards_list = []
                for ptxt in all_prompts_text:
                    cids, rew = self.multi_turn_generation(
                        ptxt, self.model, self.processing_class, self.generation_config,
                        max_new_tokens=self.max_completion_length
                    )
                    completion_ids_list.append(cids)
                    game_rewards_list.append(rew)
            else:
                completion_ids_list = [None]*len(all_prompts_text)
                game_rewards_list = [0.0]*len(all_prompts_text)

            completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)
            game_rewards_list = broadcast_object_list(game_rewards_list, from_process=0)

            game_rewards_tensor = torch.tensor(game_rewards_list, dtype=torch.float32, device=device)

            slice_ = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1)*len(prompts)
            )
            completion_ids_list = completion_ids_list[slice_]

            completion_ids_tensors = [torch.tensor(x, device=device) for x in completion_ids_list]
            completion_ids_padded = pad(completion_ids_tensors, padding_value=self.processing_class.pad_token_id)

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
        else:
            raise NotImplementedError("This example requires use_vllm=True")

        is_eos = completion_ids_padded == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids_padded.size(1)

        with torch.inference_mode(), torch.amp.autocast(
            device_type='cuda',
            dtype=(torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION','fp16')=='fp16'
                   else torch.bfloat16)
        ) if not torch.is_autocast_enabled('cuda') else nullcontext():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        completions_text = self.processing_class.batch_decode(completion_ids_padded, skip_special_tokens=True)

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i,(reward_func,reward_proc_class) in enumerate(zip(self.reward_funcs,self.reward_processing_classes)):
            keys = [k for k in inputs[0] if k not in ["prompt","completion"]]
            reward_kwargs = {k: [ex[k] for ex in inputs] for k in keys}
            out_rews = reward_func(prompts=prompts, completions=completions_text, **reward_kwargs)
            rewards_per_func[:,i] = torch.tensor(out_rews, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        game_rewards_tensor = gather(game_rewards_tensor)

        extended = torch.cat([rewards_per_func, game_rewards_tensor.unsqueeze(1)], dim=1)

        if not hasattr(self, 'reward_weights'):
            self.reward_weights = torch.ones(1, device=device)
        game_weight = torch.tensor([1.0], device=device)
        new_weights = torch.cat([self.reward_weights.to(device), game_weight])

        final_rewards = (extended * new_weights.unsqueeze(0)).sum(dim=1)
        mg = final_rewards.view(-1,self.num_generations).mean(dim=1)
        sg = final_rewards.view(-1,self.num_generations).std(dim=1)

        mg = mg.repeat_interleave(self.num_generations, dim=0)
        sg = sg.repeat_interleave(self.num_generations, dim=0)
        advantages = (final_rewards - mg)/(sg + 1e-4)

        local_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1)*len(prompts)
        )
        advantages = advantages[local_slice]

        self._metrics["rewards/game_reward"].append(game_rewards_tensor.mean().item())
        self._metrics["reward"].append(final_rewards.mean().item())
        self._metrics["reward_std"].append(sg.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids_padded,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

########################################
# 4. Reward Stub & Build Dataset
########################################

def doctor_game_reward(prompts, completions, **kwargs) -> list[float]:
    """Stub that always returns 0, the real reward is from multi_turn_generation."""
    return [0.0]*len(prompts)

def generate_initial_symptom(condition):
    """
    Generates an initial symptom description for a patient with the given condition.
    Uses OpenAI API if available, otherwise returns a generic symptom.
    """
    global client
    if client:
        try:
            prompt = (
                f"You are a patient with {condition}. "
                "Describe your main symptoms in a single sentence, as if you're "
                "initially talking to a doctor. Don't mention the diagnosis explicitly."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating initial symptom: {e}")
            return "I've been feeling unwell recently. Can you help me?"
    else:
        # Default symptoms if no OpenAI API is available
        if "flu" in condition.lower() or "influenza" in condition.lower():
            return "I have a fever, body aches, and I'm feeling very tired. Can you help?"
        elif "cold" in condition.lower():
            return "I have a runny nose, sore throat, and slight cough. What could it be?"
        elif "strep" in condition.lower():
            return "My throat is extremely sore and I have a fever. What's wrong with me?"
        elif "covid" in condition.lower():
            return "I've lost my sense of taste and smell, and I have a cough. Is this serious?"
        elif "allerg" in condition.lower():
            return "My nose is constantly running and my eyes are itchy. What's happening?"
        elif "migraine" in condition.lower():
            return "I have this intense, throbbing headache and light is bothering me."
        else:
            return "I'm not feeling well and have some concerning symptoms. Can you help me?"

def create_doctor_database():
    """
    Create a single-row dataset with system and user messages in the 'prompt' field.
    The specific symptom will be generated during training based on the condition.
    """
    row = {
        "prompt": [
            {"role":"system","content":SYSTEM_PROMPT},
            # We'll use a placeholder here, the actual patient message
            # will be generated during training in the multi_turn_generation method
            {"role":"user","content":"[PATIENT_SYMPTOMS]"}
        ],
        "answer": ""
    }
    return [row]

########################################
# 5. Configure & Train
########################################

# Make sure the GRPOConfig now includes the openai_api_key flag.
config = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    temperature=0.9,
    logging_steps=1,
    max_steps=5000,       # just a small demo
    save_steps=10,
    max_prompt_length=max_seq_length-512,
    max_completion_length=512,
    num_generations=5,  # generate 5 completions per scenario => better advantage
    output_dir=f"{save_path}/outputs",
)
import os

# Get API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable not set. Patient simulation will use default conditions.")

config.openai_api_key = api_key

# Initialize OpenAI client with the API key from config
initialize_openai_client(config.openai_api_key)
df = pd.DataFrame(create_doctor_database())
train_dataset = Dataset.from_pandas(df)

# Pass a lambda to provide the API key when instantiating the game.
trainer = DoctorGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[doctor_game_reward],
    args=config,
    train_dataset=train_dataset,
    game_object=lambda: DoctorGame(openai_api_key=config.openai_api_key)
)

print("Starting training ...")
trainer.train()

# Save final LoRA & checkpoint
model.save_lora(f"{save_path}/doctor_grpo_saved_lora")
cp_path = f"{save_path}/doctor_checkpoint"
trainer.save_model(cp_path)
trainer.state.save_to_json(f"{cp_path}/trainer_state.json")
model.save_lora(f"{save_path}/doctor_final_lora")

print("Training complete!")