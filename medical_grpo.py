# -*- coding: utf-8 -*-
"""Doctor–Patient GRPO with Unsloth"""

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
from datasets import Dataset
import torch
import pandas as pd
import re
from collections import deque

# For partial-credit scoring, etc.
import random

########################################
# 1. LOADING THE DOCTOR MODEL w/ LoRA
########################################

model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Example base model
max_seq_length = 2048
lora_rank = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,         # 4-bit quant
    fast_inference = True,       # vLLM backend
    max_lora_rank = lora_rank,   # must be >= r
    gpu_memory_utilization=0.5,
)

# Prepare LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 1234,
)

# Optional sampling parameters
from vllm import SamplingParams
temperature = 0.7
sampling_params = SamplingParams(
    temperature = temperature,
    top_p = 0.9,
    max_tokens = 256,
)

########################################
# 2. SCENARIO: DOCTOR–PATIENT
########################################

"""
We'll define a small class that simulates a single conversation episode.

We won't rely on a real GPT-4 or GPT-3.5 for the Patient, but you CAN swap in
a real LLM call to mimic the patient. 
For simplicity, let's pick from a small set of diseases randomly.

We also define a method for the Doctor to produce a turn with <reason> blocks.
Finally, we define the reward step that checks the final diagnosis (if any),
computes partial credit, etc.
"""

# We'll define a small set of possible diseases
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
    Remove <reason>...</reason> blocks from the Doctor's full text,
    leaving only the visible text that the Patient can see.
    """
    return re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL)


def parse_final_diagnosis(doctor_visible_text: str) -> str:
    """
    If the Doctor's visible text contains "Final diagnosis: XYZ",
    return "XYZ" (the text after the colon).
    Otherwise return empty string or None.
    """
    match = re.search(r"Final\s*diagnosis:\s*(.*)", doctor_visible_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


class DoctorGame:
    """
    Multi-turn conversation scenario:
    - The Patient picks a hidden disease.
    - We keep track of the conversation in two logs: 
        conv_with_reason (Doctor's hidden reasoning included),
        conv_no_reason   (visible only).
    - The Doctor tries to guess the disease within a fixed number of turns.
    - At the end, we do partial credit scoring, returning a float [0..1].
    """

    def __init__(self):
        # Patient picks a hidden disease
        self.hidden_disease = random.choice(COMMON_DISEASES)

        # Store the conversation as list of dicts: {"role": "doctor"|"patient", "content": ...}
        self.conv_no_reason = []
        self.conv_with_reason = []
        self.turn_count = 0
        self.done = False

    def get_patient_reply(self, doctor_visible_text: str) -> str:
        """
        Very simplified "Patient" logic. 
        In a real version, you might do a GPT call that:
          1) sees the conversation so far (without Doctor's <reason>).
          2) roleplays symptoms consistent with hidden_disease.
        Here, we just give a trivial response or short symptom text for demonstration.
        """
        # If the Doctor specifically gave a final diagnosis, we stop, no further reply.
        if "Final diagnosis:" in doctor_visible_text:
            self.done = True
            return ""

        # Example: we can store a few "typical symptom lines" for each disease
        disease_responses = {
            "Influenza":      "I have a fever, chills, muscle aches, a cough, and fatigue.",
            "Common cold":    "I have a runny nose, sneezing, slight cough, but no severe fever.",
            "Strep throat":   "My throat is really sore, painful swallowing, maybe a mild fever.",
            "COVID-19":       "I have a fever, cough, difficulty breathing, loss of taste, and fatigue.",
            "Allergic rhinitis": "I have sneezing, itchy nose, watery eyes, no fever though.",
            "Migraine":       "I have severe headache on one side, light sensitivity, some nausea.",
            "Mononucleosis":  "I feel extreme fatigue, sore throat, swollen lymph nodes, maybe mild fever."
        }
        # Return a short symptom line
        return disease_responses.get(self.hidden_disease, "I have some generic symptoms...")

    def doctor_system_prompt(self):
        """
        A system prompt reminding the Doctor to produce hidden chain-of-thought 
        and keep it out of the visible text.
        """
        return f"""System:
You are an AI Doctor. You can make at most {MAX_TURNS} total replies.
Each time you speak, you MUST include a hidden chain-of-thought 
in the format <reason> ... </reason>.

After the <reason> block, provide your visible text for the patient.

Never reveal the text within <reason>.

If by your final turn (turn {MAX_TURNS}) you haven't provided
'Final diagnosis: <Disease>', do so on that turn and end.

------

Conversation so far:
"""

    def build_doctor_prompt(self) -> str:
        """
        Build a single string prompt from conv_no_reason + system instructions.
        We let the model continue from "Doctor: " at the end.
        """
        # Build the system text
        prompt = self.doctor_system_prompt()

        # Append the visible conversation
        for turn in self.conv_no_reason:
            role = turn["role"].title()  # Doctor or Patient
            content = turn["content"]
            prompt += f"{role}: {content}\n"
        prompt += "Doctor:"  # we want the model's next line

        return prompt

    def step_doctor(self, doctor_model) -> float:
        """
        1) Build prompt for the Doctor (including system instructions + conv_no_reason).
        2) The Doctor model outputs text that includes <reason> block plus visible text.
        3) We parse out the visible text, store both versions.
        4) If 'Final diagnosis:' present, we end the episode.

        Return 0 as "reward" each step (the final reward is computed at the end).
        """
        # Create the prompt
        prompt = self.build_doctor_prompt()
        outputs = doctor_model.fast_generate([prompt], max_new_tokens=256, sampling_params=sampling_params)
        full_doctor_text = outputs[0]  # includes <reason> ... plus visible text

        # Remove reason tags to produce the visible text for the patient
        doc_visible = remove_reason_tags(full_doctor_text)

        # Store in both logs
        self.conv_with_reason.append({"role": "doctor", "content": full_doctor_text})
        self.conv_no_reason.append({"role": "doctor", "content": doc_visible})

        # Check for final diagnosis
        if "Final diagnosis:" in doc_visible:
            self.done = True

        return 0.0

    def step_patient(self) -> float:
        """
        1) We see the last Doctor visible text from conv_no_reason.
        2) The patient replies with some symptom line or final statement.
        3) Store it in both logs (the same text, no <reason>).
        """
        # If we're done, no more patient turns
        if self.done:
            return 0.0

        last_doc_visible = self.conv_no_reason[-1]["content"]
        pat_reply = self.get_patient_reply(last_doc_visible)

        # Add patient turn
        self.conv_no_reason.append({"role": "patient", "content": pat_reply})
        self.conv_with_reason.append({"role": "patient", "content": pat_reply})

        return 0.0

    def run_episode(self, doctor_model):
        """
        Simulate the entire conversation until done or MAX_TURNS. 
        After finishing, reveal disease, compute partial-credit reward.
        Return the reward.
        """
        self.turn_count = 0
        while not self.done and self.turn_count < MAX_TURNS:
            self.turn_count += 1
            self.step_doctor(doctor_model)
            if not self.done:
                self.step_patient()

        # The conversation is over. Let's compute final partial-credit reward.
        final_score = self.compute_reward()
        return final_score

    def compute_reward(self) -> float:
        """
        We'll do a partial-credit approach:
          - If the Doctor never gave a "Final diagnosis: X," reward = 0
          - If the Doctor did, compare X with the hidden disease. 
             * If exact match => 1.0
             * If partial match or 'close' => e.g. 0.6 or 0.8
             * Otherwise => 0.0
        You can also factor in reasoning, thoroughness, etc. for bigger completions.
        """
        # Retrieve final diagnosis if any
        doc_visible = ""
        for turn in self.conv_no_reason:
            if turn["role"] == "doctor":
                d = parse_final_diagnosis(turn["content"])
                if d:
                    doc_visible = d  # keep the last final diagnosis if multiple

        if not doc_visible:
            # No final diagnosis => 0
            return 0.0

        # Compare final diagnosis to the hidden disease
        guess = doc_visible.lower()
        answer = self.hidden_disease.lower()

        if guess == answer:
            return 1.0
        # do a naive partial match approach:
        if guess in answer or answer in guess:
            # e.g. "common cold" vs "cold"
            return 0.8
        else:
            # no match
            return 0.0


########################################
# 3. REWARD FUNCTION WRAPPER
########################################

def doctor_game_reward(prompts, completions, **kwargs) -> list[float]:
    """
    We define a stub reward function that returns 0.0 for each completion. 
    We'll rely on the *custom trainer* to actually run the multi-turn self-play 
    and produce the final reward.

    The reason: in a multi-turn environment, we can't do it purely from
    the static prompt -> single completion. We do it ourselves in the custom logic.

    For compatibility, we define a function that returns a placeholder 0.0 for each example.
    """
    # We will handle the *true* game reward inside our custom multi-turn method,
    # where we run the entire conversation. So return zeros here.
    return [0.0 for _ in prompts]


########################################
# 4. CUSTOM GRPO TRAINER
########################################

from trl import GRPOConfig
from trl.trainer.grpo_trainer import pad
import os
import torch
import torch.nn as nn

from accelerate.utils import broadcast_object_list, gather_object, set_seed

# If you have the local file "UnslothGRPOTrainer.py" from the TTT example:
# from UnslothGRPOTrainer import UnslothGRPOTrainer
# Otherwise, we place the same class inline here in short form:

from contextlib import nullcontext
from trl.data_utils import is_conversational, maybe_apply_chat_template


class DoctorGRPOTrainer:
    """
    A simplified example of a custom multi-turn trainer for the Doctor–Patient scenario.

    Key differences from standard TRL:
      1) We run multi-turn conversation in `_multi_turn_generation` with a custom game object (DoctorGame).
      2) We gather the final reward from the game object.
      3) We combine that with any additional reward functions if desired.

    For demonstration only. You can adapt the TicTacToe code to fit the doctor scenario.
    """

    def __init__(
        self,
        model,
        tokenizer,
        reward_funcs,
        args,
        train_dataset,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs  # e.g. [doctor_game_reward]
        self.args = args
        self.train_dataset = train_dataset

        # Convert any needed config from GRPOConfig
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.sampling_params = sampling_params  # from above
        self.state = type("", (), {})()  # mock
        self.state.global_step = 0

        # Reward weighting
        # If we want to do partial weighting of each reward function:
        #   e.g. [0.5, 0.5]. Must match len(reward_funcs).
        self.reward_weights = torch.ones(len(reward_funcs))

        # Just a demonstration, not a full trainer with all features
        self.max_steps = args.max_steps
        self.save_steps = args.save_steps

    def train(self):
        """
        Minimal pseudo-training loop:
          - Repeatedly sample from the dataset
          - For each sample, run multi-turn conversation
          - Compute reward -> advantage -> do GRPO step
          - (We won't show the full gradient steps for brevity.)
        """
        for step in range(self.max_steps):
            # sample a single example
            # or batch?
            example = self.train_dataset[random.randint(0, len(self.train_dataset)-1)]
            # The prompt is basically "start scenario"
            prompt = self._make_prompt(example)

            # We'll do multi-turn generation, obtain final reward
            completion_ids, game_reward = self._multi_turn_generation(prompt)

            # Optionally combine with other reward functions if you have them
            # For demonstration, we have just a single "doctor_game_reward" returning zeros, 
            # so let's ignore them. We'll directly use `game_reward`.
            total_reward = game_reward

            # Now do advantage-based RL update with that single reward
            # (In real code, you want multiple completions and compute average for advantage.)
            advantage = total_reward  # minus baseline, etc. (extremely simplified)

            # Then you'd call something like: self._update_model(completion_ids, advantage)
            # For brevity, we won't detail the entire forward/backward pass.
            # We'll just print something.
            if (step+1) % self.args.logging_steps == 0:
                print(f"[Step {step+1}] reward={total_reward:.3f}, advantage={advantage:.3f}")

            if (step+1) % self.save_steps == 0:
                print(f"[Step {step+1}] saving model checkpoint (LoRA).")
                self.model.save_lora(f"./doctor_lora_checkpoint_step{step+1}")

    def _make_prompt(self, example) -> str:
        """
        Convert a dataset row into an initial prompt for the Doctor.
        In many cases, the dataset might hold just a "dummy" prompt
        or scenario text. Here, we demonstrate it simply.
        """
        # We'll store a short user instruction: "Begin diagnosing me!"
        # In real usage, you might have more context in the dataset row.
        user_content = "I have come to see you for a diagnosis. Please help me figure out what's wrong."
        system_prompt = f"""System:
You are an AI Doctor. Always produce <reason>...</reason> for your hidden chain-of-thought.
Then produce visible text for the patient. If you haven't reached a final diagnosis by turn {MAX_TURNS}, 
output "Final diagnosis: <some disease>" on your last turn.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        # Convert to a single text prompt
        # We'll rely on the built-in chat template if you like
        prompt_text = maybe_apply_chat_template(messages, self.tokenizer)["prompt"]
        return prompt_text

    def _multi_turn_generation(self, prompt):
        """
        Runs the multi-turn conversation using the DoctorGame environment.
        Returns the final sequence of token IDs (for the entire conversation)
        and the final numeric reward from the scenario.
        """
        # Initialize the scenario
        scenario = DoctorGame()

        # We'll emulate a "starting conversation" by adding the user message 
        # into scenario's logs. But for simplicity, we skip that detail 
        # (the scenario actually starts blank, and the system prompt is inside 
        #  the final generate call).
        # If you want to incorporate the initial user message into scenario's conv_no_reason, do so.

        # We'll run the entire scenario: scenario.run_episode(self.model)
        # This method internally calls scenario.step_doctor -> scenario.step_patient, etc.
        # Returns final partial-credit reward
        final_reward = scenario.run_episode(self.model)

        # For the *trainer*, we typically need a final sequence of token IDs.
        # Since we're using multi-turn generation calls, let's just produce a 
        # single placeholder for completion_ids:
        completion_ids = [0, 1, 2]  # dummy (in practice you'd store all tokens if needed)

        return completion_ids, final_reward


########################################
# 5. CONFIG & DATASET
########################################

from trl import GRPOConfig

training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,
    temperature = temperature,
    logging_steps = 1,
    max_steps = 10,       # for demo
    save_steps = 5,
    max_prompt_length = 1024,
    max_completion_length = 512,
    num_generations = 1,   # We'll do 1 generation at a time in this simple demo
    output_dir = "./doctor_lora_output",
)

# Minimal dataset example: 
# We only need a placeholder row with a prompt, or no prompt, 
# because each scenario is generated at runtime.
df = pd.DataFrame([{"id": i, "text": "dummy scenario"} for i in range(10)])
train_dataset = Dataset.from_pandas(df)

########################################
# 6. TRAIN
########################################

trainer = DoctorGRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[doctor_game_reward],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Finally, save the LoRA
model.save_lora("./doctor_final_lora_checkpoint")

print("Training complete! LoRA saved.")
