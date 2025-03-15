# -*- coding: utf-8 -*-
"""
Doctor–Patient GRPO (Multi-Turn) 
Using GPT-4o-mini via the OpenAI API for both:
  - The Patient (roleplaying hidden disease),
  - The Judge (scoring the final conversation in [0..1]).
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
from contextlib import nullcontext

from transformers import GenerationConfig
from trl import GRPOConfig
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from trl import maybe_apply_chat_template
from trl.trainer.grpo_trainer import pad

# The same Unsloth trainer logic from Tic-Tac-Toe
from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothGRPOTrainer

# Import OpenAI and pull API key from environment variable
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

print("Imports complete.")

########################################
# 1. Load Doctor Model + LoRA
########################################

# Paths & config
save_path = "/content/drive/MyDrive/UnslothGRPO/doctorGPT4oExample"
os.makedirs(save_path, exist_ok=True)

max_seq_length = 2048
lora_rank = 16

doctor_model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # example base policy

print("Loading the Doctor (policy) model ...")
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

# Instead of loading GPT-4o-mini locally, we will call it via the OpenAI API.
print("Using OpenAI API for GPT-4o-mini as Patient & Judge roles ...")

########################################
# 2. OpenAI API Wrappers for GPT-4o-Mini Patient & Judge Functions
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
    """
    The system prompt for GPT-4o-mini in 'patient' mode.
    Instruct it to roleplay as if it has 'disease'. 
    Don't reveal it unless the Doctor's final diagnosis matches or asks for it.
    """
    return f"""System:
You are a patient with a hidden disease: {disease}.
Roleplay your symptoms. Do NOT reveal the disease unless the Doctor specifically says "Final diagnosis: {disease}" or directly asks for it by name.
If the Doctor keeps asking questions, answer them accordingly.
"""

def call_patient_model(conversation_visible, max_new_tokens=128, temperature=0.7):
    """
    Call GPT-4o-mini as Patient via the OpenAI API.
    The conversation_visible is a list of message dicts.
    Returns the patient's response as a string.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=conversation_visible,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def judge_system_prompt(conversation_with_reason: str, hidden_disease: str):
    """
    The system prompt for GPT-4o-mini acting as the Judge.
    Pass the full conversation (including <reason> blocks) and the hidden disease.
    """
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
    """
    Call GPT-4o-mini as Judge via the OpenAI API.
    Parses out the first float in [0..1] from the response.
    """
    system_text = judge_system_prompt(conversation_with_reason, hidden_disease)
    messages = [{"role": "system", "content": system_text}]
    response = openai.ChatCompletion.create(
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

########################################
# 3. The DoctorGame with OpenAI API for GPT-4o Patient & Judge
########################################

MAX_TURNS = 5

class DoctorGame:
    """
    Manages a multi-turn conversation:
      - A hidden disease.
      - The Doctor sees partial context.
      - The Patient is a GPT-4o-mini call via the OpenAI API.
      - Stores conv_with_reason (Doctor's <reason> included) and conv_no_reason (visible text).
      - After final turn, calls GPT-4o-mini as Judge for final reward in [0..1].
    """

    def __init__(self):
        self.hidden_disease = pick_hidden_disease()
        self.conv_with_reason = []
        self.conv_no_reason = []
        self.turn_count = 0
        self.done = False
        # System prompt for the patient (the Doctor does not see this)
        self.patient_system = patient_system_prompt(self.hidden_disease)

    def remove_reason_tags(self, text: str) -> str:
        return re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL)

    def parse_final_diagnosis(self, text: str) -> str:
        match = re.search(r"Final\s*diagnosis:\s*(.*)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def step_doctor(self, doc_text: str):
        """
        Processes Doctor's turn, storing both full text and visible text.
        Sets done if "Final diagnosis:" is present.
        """
        self.conv_with_reason.append({"role": "doctor", "content": doc_text})
        visible = self.remove_reason_tags(doc_text)
        self.conv_no_reason.append({"role": "doctor", "content": visible})
        if "Final diagnosis:" in visible:
            self.done = True

    def step_patient(self):
        """
        Calls GPT-4o-mini as Patient via the OpenAI API.
        The patient sees only the visible conversation plus a system message.
        """
        if self.done:
            return
        messages = [{"role": "system", "content": self.patient_system}] + self.conv_no_reason
        pat_text = call_patient_model(messages, max_new_tokens=128, temperature=0.7)
        self.conv_with_reason.append({"role": "patient", "content": pat_text})
        self.conv_no_reason.append({"role": "patient", "content": pat_text})

    def run_episode(self, doctor_model, doctor_system_prompt: str):
        """
        Runs the multi-turn conversation:
          - Up to MAX_TURNS or until the Doctor gives a final diagnosis.
          - After completion, calls the Judge for a final numeric score.
        """
        self.turn_count = 0
        while not self.done and self.turn_count < MAX_TURNS:
            self.turn_count += 1
            doc_input = self._build_doctor_prompt(doctor_system_prompt)
            doc_outs = doctor_model.fast_generate([doc_input], max_new_tokens=256, temperature=0.7)
            doc_text = doc_outs[0]
            self.step_doctor(doc_text)
            if not self.done:
                self.step_patient()
        return self.final_judge_reward()

    def _build_doctor_prompt(self, doctor_system_prompt: str) -> str:
        """
        Combines the doctor's system prompt with visible conversation history,
        ending with "Doctor:" as the cue for the Doctor's reply.
        """
        text = doctor_system_prompt
        for turn in self.conv_no_reason:
            text += f"{turn['role'].title()}: {turn['content']}\n"
        text += "Doctor:"
        return text

    def final_judge_reward(self) -> float:
        """
        Gathers the full conversation (with <reason> tags) and calls GPT-4o-mini as Judge.
        """
        conv_str = ""
        for turn in self.conv_with_reason:
            conv_str += f"{turn['role'].title()}: {turn['content']}\n"
        return call_judge_model(conv_str, self.hidden_disease)

########################################
# 4. Custom Trainer with multi_turn_generation
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
    """
    Overrides multi_turn_generation to run a DoctorGame with GPT-4o roles
    via the OpenAI API for the Patient and Judge.
    """

    def multi_turn_generation(self, prompt, model, tokenizer, generation_config, max_new_tokens=50, game_object=None):
        print("===== Starting a new Doctor–Patient Episode with GPT-4o API roles =====")
        scenario = DoctorGame()
        final_score = scenario.run_episode(model, DOCTOR_SYSTEM_PROMPT)
        # Return dummy token IDs since multi-turn generation isn't tokenized fully here.
        completion_ids = [0, 1, 2]
        return completion_ids, final_score

    # Other methods (like _prepare_inputs) remain as in your existing trainer.

########################################
# 5. Reward Function Stub
########################################

def doctor_game_reward_stub(prompts, completions, **kwargs) -> list[float]:
    """
    Stub reward function (the final reward comes from the Judge).
    """
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
    max_steps=40,      # small demonstration
    save_steps=20,
    max_prompt_length=1024,
    max_completion_length=512,
    num_generations=2, # multiple completions for advantage
    output_dir=f"{save_path}/outputs",
)

df = pd.DataFrame(build_dataset())
train_dataset = Dataset.from_pandas(df)

trainer = DoctorWithGpt4oTrainer(
    model=doctor_model,
    processing_class=doctor_tokenizer,
    reward_funcs=[doctor_game_reward_stub],
    args=training_args,
    train_dataset=train_dataset,
)

########################################
# 7. Train
########################################

if __name__ == "__main__":
    print("Starting training with GPT-4o API as Patient + Judge ...")
    trainer.train()

    # Save final LoRA & checkpoint
    doctor_model.save_lora(f"{save_path}/doctor_grpo_saved_lora")
    cp_path = f"{save_path}/doctor_checkpoint"
    trainer.save_model(cp_path)
    trainer.state.save_to_json(f"{cp_path}/trainer_state.json")
    doctor_model.save_lora(f"{save_path}/doctor_final_lora")

    print("Training complete!")
