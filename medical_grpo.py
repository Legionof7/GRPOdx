# -*- coding: utf-8 -*-
"""Doctor–Patient GRPO (Multi-Turn) with a base reward for any final diagnosis + 5 completions per scenario."""

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

# Import the same patched Unsloth trainer from your Tic-Tac-Toe example
# (We've placed it in unsloth_compiled_cache.UnslothGRPOTrainer)
from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothGRPOTrainer

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
      - The Doctor must produce <reason> blocks + visible text
      - We end if "Final diagnosis: X" is produced or we exceed MAX_TURNS
      - We do partial-credit AND a base reward if any final diagnosis is given
    """

    def __init__(self):
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

    def compute_reward(self, final_guess: str) -> float:
        """
        If final_guess is non-empty => base reward of 0.2
        Then partial-credit if it matches hidden disease.
        """
        if not final_guess:
            # no final diagnosis => 0
            return 0.0

        # base reward for providing ANY final diagnosis
        base = 0.2

        guess_lower = final_guess.lower()
        disease_lower = self.hidden_disease.lower()

        # if exact match => 1.0
        if guess_lower == disease_lower:
            return 1.0

        # partial match => 0.8
        if guess_lower in disease_lower or disease_lower in guess_lower:
            return 0.8

        # else just the base for "some guess"
        return base

########################################
# 3. Multi-Turn Generation
########################################

class DoctorGRPOTrainer(UnslothGRPOTrainer):
    """
    Similar to Tic-Tac-Toe's CustomGRPOTrainer, but for Doctor scenario.
    We override multi_turn_generation and parse "Final diagnosis: X".
    """

    def __init__(self, *args, game_object=None, **kwargs):
        self.game_object_factory = game_object
        super().__init__(*args, **kwargs)

    def multi_turn_generation(self, prompt, model, tokenizer, generation_config, max_new_tokens=50, game_object=None):
        print("============ Starting a new Doctor–Patient episode ============")
        game = self.game_object_factory() if self.game_object_factory else None
        if not game:
            raise ValueError("No game_object_factory provided")

        full_text = prompt
        total_reward = 0.0
        completion_ids = []

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

            # Check final diagnosis
            diag_match = re.search(r"Final\s*diagnosis:\s*(.*)", new_text, re.IGNORECASE)
            if diag_match:
                final_guess = diag_match.group(1).strip()
                game.done = True
                diag_reward = game.compute_reward(final_guess)
                total_reward += diag_reward
                print(f"Doctor final guess: {final_guess} => reward={diag_reward:.3f}")
                break

            if game.turn_count >= MAX_TURNS:
                # no final diagnosis => reward=0 for that
                print("No final diagnosis after max turns => 0 reward")
                break

            # Insert a minimal "patient" message
            patient_text = "Patient: I still have the same symptoms, please refine your diagnosis.\n"
            full_text += patient_text

        return completion_ids, total_reward

    def _prepare_inputs(self, inputs: dict):
        """
        Identical approach to your Tic-Tac-Toe code:
          - gather all prompts
          - multi_turn_generation on main process
          - parse final rewards
          - compute advantage
          - return RL data
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

        # gather prompts
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] 
            for example in inputs
        ]

        # tokenize
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

            # pad
            completion_ids_tensors = [torch.tensor(x, device=device) for x in completion_ids_list]
            completion_ids_padded = pad(completion_ids_tensors, padding_value=self.processing_class.pad_token_id)

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
        else:
            raise NotImplementedError("This example requires use_vllm=True")

        # mask after first EOS
        is_eos = completion_ids_padded == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids_padded.size(1)

        # log-likelihood vs ref model if any
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

        # decode completions
        completions_text = self.processing_class.batch_decode(completion_ids_padded, skip_special_tokens=True)

        # For the single reward function (which returns 0),
        # we gather that. Then we add the actual game_rewards_tensor
        # in the same advantage flow
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i,(reward_func,reward_proc_class) in enumerate(zip(self.reward_funcs,self.reward_processing_classes)):
            keys = [k for k in inputs[0] if k not in ["prompt","completion"]]
            reward_kwargs = {k: [ex[k] for ex in inputs] for k in keys}
            out_rews = reward_func(prompts=prompts, completions=completions_text, **reward_kwargs)
            rewards_per_func[:,i] = torch.tensor(out_rews, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        game_rewards_tensor = gather(game_rewards_tensor)

        # Combine them
        extended = torch.cat([rewards_per_func, game_rewards_tensor.unsqueeze(1)], dim=1)

        # We assume we have one function => self.reward_weights is shape (1,). We'll add one for the game reward
        if not hasattr(self, 'reward_weights'):
            self.reward_weights = torch.ones(1, device=device)
        game_weight = torch.tensor([1.0], device=device)
        new_weights = torch.cat([self.reward_weights.to(device), game_weight])

        final_rewards = (extended * new_weights.unsqueeze(0)).sum(dim=1)

        # advantage calc with self.num_generations
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

        # logging
        self._metrics["rewards/game_reward"].append(game_rewards_tensor.mean().item())
        self._metrics["reward"].append(final_rewards.mean().item())
        self._metrics["reward_std"].append(sg.mean().item())

        # Return RL data
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

def create_doctor_database():
    """
    We'll create a single row with a system+user message in 'prompt' field.
    The code uses 'prompt' just like in Tic-Tac-Toe.
    """
    row = {
        "prompt": [
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":"I have a headache and fatigue, can you help me?"}
        ],
        "answer": ""
    }
    return [row]

########################################
# 5. Configure & Train
########################################

config = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    temperature=0.7,
    logging_steps=1,
    max_steps=20,       # just a small demo
    save_steps=10,
    max_prompt_length=max_seq_length-512,
    max_completion_length=512,
    num_generations=5,  # generate 5 completions per scenario => better advantage
    output_dir=f"{save_path}/outputs",
)

df = pd.DataFrame(create_doctor_database())
train_dataset = Dataset.from_pandas(df)

trainer = DoctorGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[doctor_game_reward],
    args=config,
    train_dataset=train_dataset,
    game_object=DoctorGame
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
