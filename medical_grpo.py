# -*- coding: utf-8 -*-
"""
Doctor–Patient GRPO Example with Unsloth, logs each conversation to a CSV.
"""

###########################
# 0. Imports & Setup
###########################
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # Patch for GRPO

from unsloth import is_bfloat16_supported
import torch
import random
import re
import pandas as pd
from datasets import Dataset
import os

# TRL & utility
from trl import GRPOConfig, maybe_apply_chat_template
from accelerate.utils import broadcast_object_list, gather_object, set_seed

# For logging conversations as text
import datetime

print("Imports complete.")

###########################
# 1. Load Base Model & LoRA
###########################
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Example smaller model
max_seq_length = 2048
lora_rank = 16

print("Loading base model ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,       # 4-bit quantization
    fast_inference=True,     # Use vLLM backend
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,
)

print("Attaching LoRA ...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Example sampling params (vLLM style)
from vllm import SamplingParams
temperature = 0.7
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=0.9,
    max_tokens=256,
)

print("Model + LoRA ready.")

###########################
# 2. Doctor–Patient Game
###########################

COMMON_DISEASES = [
    "Influenza",
    "Common cold",
    "Strep throat",
    "COVID-19",
    "Allergic rhinitis",
    "Migraine",
    "Mononucleosis",
]

MAX_TURNS = 5

def remove_reason_tags(text: str) -> str:
    """
    Remove <reason>...</reason> blocks from the Doctor's text,
    leaving only the visible text that the Patient sees.
    """
    return re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL)

def parse_final_diagnosis(text: str) -> str:
    """
    If the Doctor's visible text contains "Final diagnosis: X",
    extract X. Otherwise return "".
    """
    match = re.search(r"Final\s*diagnosis:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


class DoctorGame:
    """
    Simulates a multi-turn conversation:
      - The Patient picks a hidden disease from a list.
      - The Doctor tries to figure it out within MAX_TURNS.
      - The Doctor's chain-of-thought is enclosed in <reason> tags, 
        which we store but do NOT show to the Patient.
      - We compute a final partial-credit reward in [0..1].
    """

    def __init__(self):
        self.hidden_disease = random.choice(COMMON_DISEASES)
        self.conv_no_reason = []    # conversation visible to the Patient
        self.conv_with_reason = []  # includes <reason> blocks
        self.turn_count = 0
        self.done = False

    def doctor_system_prompt(self) -> str:
        """
        A system message that instructs the Doctor to always produce <reason> blocks.
        """
        return f"""System:
You are an AI Doctor. You can make at most {MAX_TURNS} total replies.
Each time you speak, you MUST include a hidden chain-of-thought 
in the format <reason> ... </reason>.

After that, provide visible text for the patient.

If by your final turn (turn {MAX_TURNS}) you haven't provided
"Final diagnosis: <Disease>", do so and then stop.

Conversation so far:
"""

    def build_doctor_prompt(self) -> str:
        """
        We combine the system instructions with the visible conversation so far,
        and end with "Doctor:". The LLM is expected to continue from there.
        """
        prompt_text = self.doctor_system_prompt()
        for turn in self.conv_no_reason:
            role = turn["role"].title()  # "Doctor" or "Patient"
            content = turn["content"]
            prompt_text += f"{role}: {content}\n"
        prompt_text += "Doctor:"
        return prompt_text

    def get_patient_reply(self, doctor_visible_text: str) -> str:
        """
        Minimal 'Patient' logic. We pick a short symptom text based on the hidden disease.
        In a real system, you'd do a model call or more advanced logic.
        """
        if "Final diagnosis:" in doctor_visible_text:
            self.done = True
            return ""

        # disease-based simple response
        disease_map = {
            "Influenza": "I have a fever, chills, cough, and muscle aches.",
            "Common cold": "I have a runny nose, sneezing, maybe a mild cough.",
            "Strep throat": "I have a very sore throat and difficulty swallowing.",
            "COVID-19": "I have fever, cough, and loss of taste/smell.",
            "Allergic rhinitis": "I have itchy eyes, runny nose, but no fever.",
            "Migraine": "I have a severe headache, light sensitivity, and some nausea.",
            "Mononucleosis": "I have fatigue, sore throat, and swollen lymph nodes."
        }
        return disease_map.get(self.hidden_disease, "I have some generic symptoms.")

    def step_doctor(self, doctor_model) -> None:
        """
        1) Build the Doctor's prompt with system instructions + conv_no_reason.
        2) The Doctor model outputs <reason>...</reason> + visible text.
        3) We store both versions.
        4) If the Doctor says "Final diagnosis: ...", we set done=True.
        """
        prompt = self.build_doctor_prompt()
        outs = doctor_model.fast_generate([prompt], sampling_params=sampling_params)
        # For vLLM, outs is a list of RequestOutput objects. We want the text:
        full_doctor_text = outs[0].outputs[0].text  # includes <reason> plus visible text

        doc_visible = remove_reason_tags(full_doctor_text)

        self.conv_with_reason.append({"role": "doctor", "content": full_doctor_text})
        self.conv_no_reason.append({"role": "doctor", "content": doc_visible})

        if "Final diagnosis:" in doc_visible:
            self.done = True

    def step_patient(self) -> None:
        """
        The Patient sees the last Doctor visible text, replies with symptom info,
        or ends if we've got a final diagnosis.
        """
        if self.done:
            return
        last_doc_visible = self.conv_no_reason[-1]["content"]
        pat_msg = self.get_patient_reply(last_doc_visible)

        self.conv_no_reason.append({"role": "patient", "content": pat_msg})
        self.conv_with_reason.append({"role": "patient", "content": pat_msg})

    def run_episode(self, doctor_model) -> float:
        """
        Main conversation loop up to MAX_TURNS or final diagnosis.
        Then compute partial-credit reward in [0..1].
        Return that final reward, plus the conversation logs.
        """
        while not self.done and self.turn_count < MAX_TURNS:
            self.turn_count += 1
            self.step_doctor(doctor_model)
            if not self.done:
                self.step_patient()

        final_reward = self.compute_reward()
        return final_reward

    def compute_reward(self) -> float:
        """
        Extract final diagnosis if given, compare to hidden disease, return partial credit.
        """
        final_guess = ""
        for turn in self.conv_no_reason:
            if turn["role"] == "doctor":
                guess = parse_final_diagnosis(turn["content"])
                if guess:
                    final_guess = guess  # keep last if multiple

        if not final_guess:
            # no final diagnosis => 0
            return 0.0

        guess_lower = final_guess.lower()
        disease_lower = self.hidden_disease.lower()
        if guess_lower == disease_lower:
            return 1.0
        # partial match?
        if guess_lower in disease_lower or disease_lower in guess_lower:
            return 0.8
        return 0.0


###########################
# 3. Reward Function Stub
###########################
def doctor_game_reward(prompts, completions, **kwargs) -> list[float]:
    """
    A stub returning 0.0 for each text. We'll rely on the custom 
    multi-turn logic to compute the real reward. 
    This is just to satisfy TRL's interface if needed.
    """
    return [0.0] * len(prompts)


###########################
# 4. Custom Trainer w/ Logging
###########################
class DoctorGRPOTrainer:
    """
    - For each training step, runs a multi-turn "DoctorGame"
    - Gathers the final reward
    - Logs the conversation details
    - (Normally you'd do advantage-based RL updates)
    """

    def __init__(self, model, tokenizer, reward_funcs, args, train_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs  # e.g. [doctor_game_reward]
        self.args = args
        self.train_dataset = train_dataset
        self.state = type("", (), {})()
        self.state.global_step = 0

        self.max_steps = args.max_steps
        self.save_steps = args.save_steps
        self.logging_steps = args.logging_steps

        # We'll store each conversation in memory to log at the end
        self.episodes_log = []

    def _make_prompt(self, example) -> str:
        """
        Convert a dataset row into an initial prompt for the Doctor.
        Must pass a dict with key "messages" to maybe_apply_chat_template
        so it returns a combined "prompt".
        """
        system_prompt = """System:
You are an AI Doctor. You must produce <reason>...</reason> 
for hidden reasoning, then visible text. 
If you haven't provided a 'Final diagnosis:' by the last turn, 
do so.
"""
        user_content = "I feel sick and need help."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        conversation_dict = {"messages": messages}
        out = maybe_apply_chat_template(conversation_dict, self.tokenizer)
        if "prompt" in out:
            prompt_text = out["prompt"]
        elif "text" in out:
            prompt_text = out["text"]
        else:
            raise ValueError("maybe_apply_chat_template did not return 'prompt' or 'text'.")
        return prompt_text

    def train(self):
        """
        Simplified training loop:
         - We'll do self.max_steps steps
         - Sample from train_dataset
         - Build prompt
         - Run multi-turn generation
         - Log conversation & final reward
         - (Real code: advantage-based RL updates)
        """
        for step in range(self.max_steps):
            # pick random row
            ex = self.train_dataset[random.randint(0, len(self.train_dataset) - 1)]
            prompt = self._make_prompt(ex)

            # multi-turn conversation
            scenario = DoctorGame()
            final_reward = scenario.run_episode(self.model)

            # Store conversation logs
            # scenario.conv_with_reason: full text including <reason>
            # scenario.conv_no_reason: only visible text
            # scenario.hidden_disease
            # final_reward

            # Flatten them into strings
            conv_with_reason_str = "\n".join(
                f"{turn['role'].title()}: {turn['content']}" 
                for turn in scenario.conv_with_reason
            )
            conv_no_reason_str = "\n".join(
                f"{turn['role'].title()}: {turn['content']}" 
                for turn in scenario.conv_no_reason
            )

            self.episodes_log.append({
                "step": step+1,
                "hidden_disease": scenario.hidden_disease,
                "reward": final_reward,
                "conversation_with_reason": conv_with_reason_str,
                "conversation_no_reason": conv_no_reason_str
            })

            # Print for quick view
            print(f"[Step {step+1}] final reward = {final_reward:.3f}")

            if (step+1) % self.save_steps == 0:
                print(f"[Step {step+1}] saving LoRA checkpoint.")
                self.model.save_lora(f"./doctor_lora_checkpoint_step{step+1}")

            self.state.global_step += 1

        print("Training finished!")

        # final save
        self.model.save_lora("./doctor_final_lora_checkpoint")

        # Also log to a CSV for later inspection
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_df = pd.DataFrame(self.episodes_log)
        os.makedirs("./logs", exist_ok=True)
        csv_path = f"./logs/doctor_episodes_{timestamp}.csv"
        logs_df.to_csv(csv_path, index=False)
        print(f"Saved conversation logs to {csv_path}")


###########################
# 5. Configure & Build Trainer
###########################
from trl import GRPOConfig

training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    temperature=temperature,
    logging_steps=1,
    max_steps=10,        # short for demo
    save_steps=5,
    max_prompt_length=1024,
    max_completion_length=512,
    num_generations=1,   # minimal
    output_dir="./doctor_lora_output",
)

# Minimal dataset with dummy rows
df = pd.DataFrame({"dummy": [f"row {i}" for i in range(5)]})
train_dataset = Dataset.from_pandas(df)

trainer = DoctorGRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[doctor_game_reward],
    args=training_args,
    train_dataset=train_dataset,
)

###########################
# 6. Train
###########################
if __name__ == "__main__":
    trainer.train()
    print("All done!")
