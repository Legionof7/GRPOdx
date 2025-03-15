# -*- coding: utf-8 -*-
"""
Doctor–Patient GRPO (Multi-Turn) 
Using GPT-4o-mini via the OpenAI API for both:
  - The Patient (roleplaying hidden disease),
  - The Judge (scoring the final conversation in [0..1]).

Fixed: Now we do not pass 'sampling_params' or 'max_new_tokens';
instead we use 'max_tokens' (which is accepted by fast_generate).
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
import datetime
import logging
from contextlib import nullcontext

from transformers import GenerationConfig
from trl import GRPOConfig
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from trl import maybe_apply_chat_template
from trl.trainer.grpo_trainer import pad

# The same Unsloth trainer logic from Tic-Tac-Toe
from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothGRPOTrainer

# Import OpenAI and set up API key handling
import openai
from openai import OpenAI

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/training_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting medical GRPO training session")
logger.info("Imports complete.")

########################################
# 1. Load Doctor Model + LoRA
########################################

save_path = "./doctor_outputs"
os.makedirs(save_path, exist_ok=True)

max_seq_length = 2048
lora_rank = 16

doctor_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

logger.info(f"Loading the Doctor (policy) model: {doctor_model_name}")
doctor_model, doctor_tokenizer = FastLanguageModel.from_pretrained(
    model_name=doctor_model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,
)

doctor_model = FastLanguageModel.get_peft_model(
    doctor_model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
logger.info(f"Doctor model loaded with LoRA rank {lora_rank}")

logger.info("Using OpenAI API for GPT-4o-mini as Patient & Judge roles ...")

########################################
# 2. GPT-4o Patient & Judge Functions
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

def pick_hidden_disease():
    return random.choice(COMMON_DISEASES)

def patient_system_prompt(disease: str):
    return f"""System:
You are a patient with a hidden disease: {disease}.
Roleplay your symptoms. Do NOT reveal the disease unless the Doctor specifically says "Final diagnosis: {disease}" or directly asks for it by name.
If the Doctor keeps asking questions, answer them accordingly.
"""

def call_patient_model(conversation_visible, max_new_tokens=128, temperature=0.7):
    print(f"\n🔄 OPENAI API CALL - Patient Role")
    if not openai.api_key:
        print("❌ ERROR: No OpenAI API key provided. Returning dummy text.")
        return "I'm missing my API key, sorry!"
    try:
        openai_messages = []
        system_message = next((msg for msg in conversation_visible if msg["role"] == "system"), None)
        if system_message:
            openai_messages.append({"role": "system", "content": system_message["content"]})
        for msg in conversation_visible:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "doctor":
                openai_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "patient":
                openai_messages.append({"role": "assistant", "content": msg["content"]})
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"❌ ERROR in call_patient_model: {str(e)}")
        return f"(Error calling GPT-4o-mini: {str(e)})"

def judge_system_prompt(conversation_with_reason: str, hidden_disease: str):
    return f"""System:
You are a medical conversation evaluator (the Judge).
Below is the entire conversation (including the Doctor's <reason> blocks).
The hidden disease the Patient had was: {hidden_disease}

Please rate how well the Doctor performed on:
1) Asking relevant questions
2) Providing correct or near-correct final diagnosis
3) Clarity and logical reasoning in the <reason> blocks

Return a SINGLE float in [0..1] (the FIRST float you mention),
where 0 means extremely poor, 1 means excellent.

Conversation:
{conversation_with_reason}

Now give me the single float:
"""

def call_judge_model(conversation_with_reason: str, hidden_disease: str, max_new_tokens=64, temperature=0.0):
    print(f"\n🔄 OPENAI API CALL - Judge Role")
    if not openai.api_key:
        print("❌ ERROR: No OpenAI API key provided. Returning default 0.0")
        return 0.0
    system_text = judge_system_prompt(conversation_with_reason, hidden_disease)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "Please evaluate this conversation and provide a single score between 0 and 1."}
    ]
    try:
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        raw_judge = response.choices[0].message.content.strip()
        match = re.search(r"\b0(\.\d+)?\b|\b1(\.0+)?\b", raw_judge)
        if match:
            val = float(match.group(0))
            return max(0.0, min(1.0, val))
        return 0.0
    except Exception as e:
        print(f"❌ ERROR calling GPT-4o for judge: {str(e)}")
        return 0.0

########################################
# 3. DoctorGame for multi-turn conversation
########################################

MAX_TURNS = 5

class DoctorGame:
    def __init__(self):
        self.hidden_disease = pick_hidden_disease()
        self.conv_with_reason = []
        self.conv_no_reason = []
        self.turn_count = 0
        self.done = False
        self.patient_system = patient_system_prompt(self.hidden_disease)

    def remove_reason_tags(self, text: str) -> str:
        return re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL)

    def parse_final_diagnosis(self, text: str) -> str:
        match = re.search(r"Final\s*diagnosis:\s*(.*)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def step_doctor(self, doc_text: str):
        self.conv_with_reason.append({"role": "doctor", "content": doc_text})
        visible = self.remove_reason_tags(doc_text)
        self.conv_no_reason.append({"role": "doctor", "content": visible})
        if "Final diagnosis:" in visible:
            self.done = True

    def step_patient(self):
        if self.done:
            return
        openai_messages = [{"role": "system", "content": self.patient_system}]
        for msg in self.conv_no_reason:
            if msg["role"] == "doctor":
                openai_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "patient":
                openai_messages.append({"role": "assistant", "content": msg["content"]})
        pat_text = call_patient_model(openai_messages, max_new_tokens=128, temperature=0.7)
        self.conv_with_reason.append({"role": "patient", "content": pat_text})
        self.conv_no_reason.append({"role": "patient", "content": pat_text})

    def run_episode(self, doctor_model, doctor_system_prompt: str):
        self.turn_count = 0
        self.done = False
        self.conv_with_reason = []
        self.conv_no_reason = []

        while not self.done and self.turn_count < MAX_TURNS:
            self.turn_count += 1
            doc_input = self._build_doctor_prompt(doctor_system_prompt)
            print(f"Generating doctor response for turn {self.turn_count}...")

            # Use fast_generate with standard arguments and 'max_tokens' instead of 'max_new_tokens'
            doc_outs = doctor_model.fast_generate(
                [doc_input],
                max_tokens=256,
                temperature=0.7,
            )

            if hasattr(doc_outs[0], "outputs"):
                doc_text = doc_outs[0].outputs[0].text
            else:
                doc_text = doc_outs[0]
            print(f"Doctor response: {doc_text[:60]}...")
            self.step_doctor(doc_text)
            if not self.done:
                print(f"--- Now getting patient response ---")
                self.step_patient()

        conv_str = ""
        for turn in self.conv_with_reason:
            conv_str += f"{turn['role'].title()}: {turn['content']}\n"
        reward = call_judge_model(conv_str, self.hidden_disease)
        print(f"Judge gave reward: {reward}")
        return reward

    def _build_doctor_prompt(self, doctor_system_prompt: str) -> str:
        text = doctor_system_prompt
        for turn in self.conv_no_reason:
            text += f"{turn['role'].title()}: {turn['content']}\n"
        text += "Doctor:"
        return text

########################################
# 4. Custom Trainer: override _prepare_inputs
########################################

DOCTOR_SYSTEM_PROMPT = """
System:
You are an AI Doctor. Each time you speak, you MUST include a hidden chain-of-thought
in the format <reason> ... </reason>. Then provide the visible text for the patient.

If by your final turn you haven't said: "Final diagnosis: XYZ", do so and end.

Possible diseases:
- Influenza
- Common cold
- Strep throat
- ...
"""

class DoctorWithGpt4oTrainer(UnslothGRPOTrainer):
    def multi_turn_generation(self, prompt, model, tokenizer, generation_config, max_tokens=50, game_object=None):
        print("===== STARTING Doctor–Patient Episode with GPT-4o API =====")
        scenario = DoctorGame()
        final_reward = scenario.run_episode(model, DOCTOR_SYSTEM_PROMPT)
        completion_ids = [0, 1, 2]
        return completion_ids, final_reward

    def _prepare_inputs(self, inputs: dict):
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
        prompts_text = []
        for example in inputs:
            text = ""
            for msg in example["prompt"]:
                text += f"{msg['role'].title()}: {msg['content']}\n"
            text += "Doctor:"
            prompts_text.append(text)

        prompt_ids = torch.tensor([[0]], device=device)
        prompt_mask = torch.tensor([[1]], device=device)

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
                        max_tokens=self.max_completion_length
                    )
                    completion_ids_list.append(cids)
                    game_rewards_list.append(rew)
            else:
                completion_ids_list = [None] * len(all_prompts_text)
                game_rewards_list = [0.0] * len(all_prompts_text)

            completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)
            game_rewards_list = broadcast_object_list(game_rewards_list, from_process=0)

            game_rewards_tensor = torch.tensor(game_rewards_list, dtype=torch.float32, device=device)
            from trl.trainer.grpo_trainer import pad
            completion_ids_tensors = [torch.tensor(x, device=device) for x in completion_ids_list]
            completion_ids_padded = pad(completion_ids_tensors, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
        else:
            raise NotImplementedError("Must use use_vllm=True for multi-turn in this snippet.")

        batch_size = len(inputs)
        completion_length = completion_ids_padded.size(1)
        ref_per_token_logps = torch.zeros(batch_size, completion_length, device=device)
        game_rewards_tensor = gather(game_rewards_tensor)

        final_rewards = game_rewards_tensor
        mg = final_rewards.mean()
        sg = final_rewards.std() if final_rewards.size(0) > 1 else 1.0
        advantages = (final_rewards - mg) / (sg + 1e-4)

        prompt_mask = torch.ones(batch_size, 1, dtype=torch.int, device=device)
        completion_mask = torch.ones(batch_size, completion_length, dtype=torch.int, device=device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        self._metrics["rewards/game_reward"].append(final_rewards.mean().item())
        self._metrics["reward"].append(final_rewards.mean().item())
        self._metrics["reward_std"].append(final_rewards.std().item() if final_rewards.size(0) > 1 else 0.0)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids_padded,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

########################################
# 5. Reward Function Stub
########################################

def doctor_game_reward_stub(prompts, completions, **kwargs) -> list[float]:
    return [0.0] * len(prompts)

########################################
# 6. Minimal Training Setup
########################################

def build_dataset():
    row = {
        "prompt": [
            {"role": "system", "content": "You are an AI Doctor. Provide a diagnosis eventually."},
            {"role": "user", "content": "I have a headache, any ideas?"}
        ],
        "answer": ""
    }
    return [row]

training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    temperature=0.7,
    logging_steps=5,
    max_steps=10,      # short demonstration
    save_steps=5,
    max_prompt_length=1024,
    max_completion_length=512,
    num_generations=2,  # multiple completions for advantage
    output_dir="./doctor_outputs/checkpoints",
)

df = pd.DataFrame(build_dataset())
train_dataset = Dataset.from_pandas(df)

def create_trainer(model, tokenizer):
    trainer = DoctorWithGpt4oTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[doctor_game_reward_stub],
        args=training_args,
        train_dataset=train_dataset,
    )
    return trainer

########################################
# 7. Run
########################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a doctor model with GPT-4o via OpenAI API")
    parser.add_argument("--openai_api_key", type=str, default="")
    args = parser.parse_args()

    if args.openai_api_key:
        openai.api_key = args.openai_api_key
        print("✅ OpenAI API key set from command line argument")
    elif not openai.api_key:
        print("❌ No OpenAI API key. Please provide with --openai_api_key. Exiting.")
        exit(1)

    trainer = create_trainer(doctor_model, doctor_tokenizer)
    trainer.train()

    # Save final
    lora_path = "./doctor_lora"
    doctor_model.save_lora(lora_path)
    print(f"Saved LoRA to {lora_path}")
