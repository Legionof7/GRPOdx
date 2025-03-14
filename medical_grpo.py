# -*- coding: utf-8 -*-
"""Doctor–Patient GRPO (Multi-Turn) using the same Tic-Tac-Toe code structure."""

########################################
# 0. Imports & Setup
########################################

# EXACT same imports from the Tic-Tac-Toe notebook
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
from datasets import Dataset
import torch
import pandas as pd

import random
import re
import os
import math
import textwrap
import warnings
from collections import deque

from transformers import TrainerState
from trl import GRPOConfig, GRPOTrainer
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import maybe_apply_chat_template, apply_chat_template
from trl.import_utils import is_rich_available
from trl.data_utils import is_conversational
from trl.trainer.grpo_trainer import pad
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from contextlib import nullcontext

# The patched Unsloth trainer from your Tic-Tac-Toe example
from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothGRPOTrainer

print("Imports complete.")

########################################
# 1. Model Loading (Same as TTT Example)
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
# 2. Define the Doctor–Patient Scenario
########################################

# We'll define a small set of diseases to pick from
COMMON_DISEASES = [
    "Influenza",
    "Common cold",
    "Strep throat",
    "COVID-19",
    "Allergic rhinitis",
    "Migraine",
    "Mononucleosis",
]

# System prompt: we must produce <reason> ... </reason> plus a final diagnosis
SYSTEM_PROMPT = """
You are a Doctor diagnosing a patient. Always provide:
<reason> your chain-of-thought reasoning here </reason>
Then provide short visible text for the patient.

When you conclude, provide a final line like:
Final diagnosis: XYZ
"""

MAX_TURNS = 5

class DoctorGame:
    """
    Multi-turn scenario:
      - Hidden disease chosen randomly
      - The Doctor must produce <reason> blocks + visible text each turn
      - The conversation ends if the Doctor says "Final diagnosis: X" 
        or we exceed MAX_TURNS
      - Partial-credit reward if final guess is correct or close
    """

    def __init__(self):
        self.hidden_disease = random.choice(COMMON_DISEASES)
        self.turn_count = 0
        self.done = False

        # We'll store the conversation text (both full + reason, and visible only)
        # so we can debug or finalize. For RL we might not strictly need them.
        self.conv_with_reason = []
        self.conv_no_reason = []

    def is_done(self):
        return self.done or self.turn_count >= MAX_TURNS

    def parse_final_diagnosis(self, text: str) -> str:
        """
        If there's "Final diagnosis: X" in the visible text, 
        return "X". Otherwise return "".
        """
        match = re.search(r"Final\s*diagnosis:\s*(.*)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def compute_reward(self, final_guess: str) -> float:
        """
        Compare final_guess to hidden_disease, return partial credit in [0..1].
        """
        if not final_guess:
            return 0.0
        guess_lower = final_guess.lower()
        disease_lower = self.hidden_disease.lower()
        if guess_lower == disease_lower:
            return 1.0
        if guess_lower in disease_lower or disease_lower in guess_lower:
            # partial match 
            return 0.8
        return 0.0

########################################
# 3. External Action & multi_turn_generation
########################################

# We define an "action" for the Doctor each turn, which might be a guess or more Q&A.
# We'll parse out "Final diagnosis: X" or see if the user kept talking.

def perform_external_action(action_text, game):
    """
    If the Doctor provides "Final diagnosis: XXX", we finalize the conversation.
    Otherwise, we can keep going. For partial credit, we return the reward at the end. 
    But we won't do anything 'turn-based' except check if the doc ended.

    We'll parse "Final diagnosis:" from the action_text if present. 
    Return (some_summary_text, reward).
    """
    # Check if there's a final diagnosis in the text
    final_guess = game.parse_final_diagnosis(action_text)
    if final_guess:
        # conversation ends
        game.done = True
        reward = game.compute_reward(final_guess)
        return (f"Diagnosis guess: {final_guess}", reward)
    else:
        # no final diagnosis => 0 reward for now, continue
        return ("Still not final diagnosis", 0.0)


class CustomGRPOTrainer(UnslothGRPOTrainer):
    """
    This class parallels the Tic-Tac-Toe version. We override multi_turn_generation
    to handle the Doctor scenario. 
    We'll parse out <reason> tags or final diagnosis from the text each turn.
    """

    def __init__(self, *args, game_object=None, **kwargs):
        self.game_object_factory = game_object
        super().__init__(*args, **kwargs)
        self.num_iterations = 1

    def multi_turn_generation(self, prompt, model, tokenizer, generation_config, max_new_tokens=50, game_object=None):
        """
        Called once per sample in `_prepare_inputs`. We run the multi-turn scenario:
          - The Doctor sees the prompt (system + user).
          - The Doctor outputs text with <reason>...
          - We check if there's "Final diagnosis:" => done
          - If not done, we ask for next turn, up to MAX_TURNS
        Returns (completion_ids, total_reward).
        """
        print("============ Starting a new Doctor–Patient episode ============")

        # game scenario
        game = self.game_object_factory() if self.game_object_factory else None
        if not game:
            raise ValueError("No game_object_factory provided")

        device = next(model.parameters()).device
        full_text = prompt
        total_reward = 0.0
        completion_ids = []

        # Keep going up to MAX_TURNS or until done
        while not game.is_done():
            # Generate next Doctor response
            outputs = self.llm.generate(
                [full_text],
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            new_text = outputs[0].outputs[0].text
            new_token_ids = outputs[0].outputs[0].token_ids

            # add to our overall text & token list
            full_text += new_text
            completion_ids.extend(new_token_ids)
            game.turn_count += 1

            # parse final diagnosis
            # We'll treat everything as if it's the Doctor's latest message
            diag_match = re.search(r"Final\s*diagnosis:\s*(.*)", new_text, re.IGNORECASE)
            if diag_match:
                final_guess = diag_match.group(1).strip()
                # conversation ends
                game.done = True
                # partial-credit
                diag_reward = game.compute_reward(final_guess)
                total_reward += diag_reward
                print(f"Doctor final guess = {final_guess} => reward={diag_reward}")
                break

            if game.turn_count >= MAX_TURNS:
                print("Reached max turns without final diagnosis => reward=0")
                break

            # If there's no final diagnosis, we do an intermediate step reward of 0
            # (You could add shaping if you want.)
            # Then we could add "Patient" response or some feedback to the prompt if desired
            # For simplicity, we'll just proceed with the same prompt. 
            # Or add a line like "Patient: 'I still have a fever...'"
            # to keep the multi-turn going. For brevity, we keep it minimal.

            # If we want the conversation to proceed, we append a short patient line:
            patient_text = "Patient: I'm still describing my symptoms. Could you refine the diagnosis?"
            full_text += "\n" + patient_text + "\n"

        return completion_ids, total_reward

    def _prepare_inputs(self, inputs: dict):
        """
        Almost identical to the TTT version. We do:
         - tokenize prompt
         - call multi_turn_generation(...) in the main process
         - gather the final reward
         - compute advantage
         - return the data needed for the forward pass (which triggers the RL update).
        """
        if not hasattr(self, "generation_config"):
            from transformers import GenerationConfig
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

        # The next part is standard: we gather the "prompt" from each example, apply chat template
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
            # Move model to vLLM if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Gather all prompts to main process
            all_prompts_text = gather_object(prompts_text)

            if self.accelerator.is_main_process:
                completion_ids_list = []
                game_rewards_list = []
                for ptxt in all_prompts_text:
                    cids, rew = self.multi_turn_generation(
                        ptxt, self.model, self.processing_class, self.generation_config,
                        max_new_tokens=self.max_completion_length,
                    )
                    completion_ids_list.append(cids)
                    game_rewards_list.append(rew)
            else:
                completion_ids_list = [None]*len(all_prompts_text)
                game_rewards_list = [0.0]*len(all_prompts_text)

            completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)
            game_rewards_list = broadcast_object_list(game_rewards_list, from_process=0)

            # convert to torch
            game_rewards_tensor = torch.tensor(game_rewards_list, dtype=torch.float32, device=device)

            # slice for local process
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids_list = completion_ids_list[process_slice]

            # pad and concat
            completion_ids_tensors = [torch.tensor(x, device=device) for x in completion_ids_list]
            completion_ids_padded = pad(completion_ids_tensors, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
        else:
            # If not using vLLM, you'd do a normal generate. Omitted for brevity
            raise NotImplementedError("Please enable use_vllm for this code block")

        # We do the rest: mask after first EOS, compute log-likelihood, do advantage
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
            # if we have a ref model
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # The completions_text
        completions_text = self.processing_class.batch_decode(
            completion_ids_padded, skip_special_tokens=True
        )
        # Build a dummy 'completions' structure
        if is_conversational(inputs[0]):
            completions = []
            for p, c in zip(prompts, completions_text):
                bootstrap = p.pop()["content"] if p[-1]["role"]=="assistant" else ""
                completions.append([{"role":"assistant","content":bootstrap+c}])
        else:
            completions = completions_text

        # we do reward function calls:
        # The single "reward_func" is a dummy, plus we have the real "game_rewards_tensor"
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i,(reward_func,reward_proc_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            # same logic as TTT
            if isinstance(reward_func, nn.Module):
                # skip
                pass
            else:
                keys = [k for k in inputs[0] if k not in ["prompt","completion"]]
                reward_kwargs = {k: [ex[k] for ex in inputs] for k in keys}
                out_rews = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:,i] = torch.tensor(out_rews, dtype=torch.float32, device=device)

        # gather across processes
        rewards_per_func = gather(rewards_per_func)
        game_rewards_tensor = gather(game_rewards_tensor)

        # sum them
        # We'll add the game reward with weight=1.0
        # If there's only one function, shape is (B,1), we add game reward as a new column
        extended = torch.cat([rewards_per_func, game_rewards_tensor.unsqueeze(1)],dim=1)
        # we have reward_weights in self.reward_weights (size e.g. 1). We expand it by 1 for the game reward
        game_weight = torch.tensor([1.0], device=device)
        new_weights = torch.cat([self.reward_weights.to(device), game_weight])
        final_rewards = (extended * new_weights.unsqueeze(0)).sum(dim=1)

        # Now do advantage across the group of completions for each prompt
        # self.num_generations might be >1, we do grouping:
        mg = final_rewards.view(-1,self.num_generations).mean(dim=1)
        sg = final_rewards.view(-1,self.num_generations).std(dim=1)

        mg = mg.repeat_interleave(self.num_generations,dim=0)
        sg = sg.repeat_interleave(self.num_generations,dim=0)
        advantages = (final_rewards - mg)/(sg+1e-4)

        local_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1)*len(prompts)
        )
        advantages = advantages[local_slice]

        # Logging
        # This is effectively the same approach used in TTT code
        self._metrics["rewards/game_reward"].append(game_rewards_tensor.mean().item())
        self._metrics["reward"].append(final_rewards.mean().item())
        self._metrics["reward_std"].append(sg.mean().item())

        # Return data that will be used for RL updates
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids_padded,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

########################################
# 4. Reward & Prompt Creation
########################################

def doctor_game_reward(prompts, completions, **kwargs) -> list[float]:
    """
    A stub reward function that always returns 0. 
    We'll rely on the actual final reward from multi_turn_generation logic.
    """
    return [0.0 for _ in prompts]

def create_doctor_database():
    """
    Minimal dataset with a single "prompt" that starts the scenario.
    """
    # We'll have just one row that references the system prompt
    # plus a user line "I feel ill" or something
    # We store the data in a shape similar to the TTT 'create_database'.
    # Because we must feed "prompt" in the trainer code
    prompt = [
        {"role":"system", "content":SYSTEM_PROMPT},
        {"role":"user",   "content":"I feel sick, help me."}
    ]
    data = [{"prompt": prompt, "answer": ""}]
    return data

########################################
# 5. Build the Trainer & Train
########################################

training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,
    temperature = 0.8,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.01,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1,
    num_generations = 2,   # do multiple completions for advantage
    max_prompt_length = max_seq_length - 512,
    max_completion_length = 512,
    max_steps = 20,  # small demo
    save_steps = 10,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = f"{save_path}/outputs",
)

df = pd.DataFrame(create_doctor_database())
train_dataset = Dataset.from_pandas(df)

trainer = CustomGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[doctor_game_reward],
    args=training_args,
    train_dataset=train_dataset,
    game_object=DoctorGame,  # pass the factory for the Doctor scenario
)

print("Starting training ...")
trainer.train()

# Save final lora
model.save_lora(f"{save_path}/doctor_grpo_saved_lora")

# Save a full checkpoint
checkpoint_path = f"{save_path}/doctor_checkpoint"
trainer.save_model(checkpoint_path)
trainer.state.save_to_json(f"{checkpoint_path}/trainer_state.json")
model.save_lora(f"{save_path}/doctor_final_lora")
print("Training complete!")
