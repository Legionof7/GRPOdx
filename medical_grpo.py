#!/usr/bin/env python3

import os
import re
import time
import random
import logging
import asyncio
from typing import List, Dict, Any

# Unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
# For the standard "Approach A" GRPO
from trl import GRPOConfig, GRPOTrainer

# We'll use "AsyncOpenAI" for calling GPT-4o-mini to generate scenario seeds + judge
from openai import AsyncOpenAI

# For the dataset
from datasets import Dataset

###############################################################################
#                      CONFIG & GLOBAL CONSTANTS
###############################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local doctor model name (on HF or local)
LOCAL_DOCTOR_MODEL_NAME = "unsloth/Phi-4"  # or any Unsloth-compatible model
# GPT-4o-mini or some other OpenAI-based small model
JUDGE_MODEL = "gpt-4o-mini"
SCENARIO_MODEL = "gpt-4o-mini"

NUM_OUTER_STEPS = 2     # how many times we gather new scenarios & train
SCENARIOS_PER_LOOP = 10 # gather 10 new scenarios each loop
MAX_TURNS = 5           # your instructions to the local model for how many doc/patient lines
LR = 5e-6               # learning rate
MAX_GLOBAL_STEPS = 100  # an upper bound for training steps

###############################################################################
#                           UTILITY FUNCTIONS
###############################################################################
async def generate_scenario_seed(openai_api_key: str) -> Dict[str, Any]:
    """
    Call GPT-4o-mini to produce a hidden disease and a short patient complaint.
    We'll store them as a single string for the 'prompt' in our dataset.
    The local model will see the instructions "Act out both patient & doctor"
    but also we feed a hidden disease that it *should* reveal only at 'Final diagnosis:' line.
    """
    # Prompt GPT-4o-mini to pick a disease + complaint
    scenario_prompt = (
        "Pick a common disease (keep it hidden), then give a short opening line from the patient. "
        "Output them in the format:\n"
        "Disease: X\n"
        "Complaint: Y\n"
        "But do NOT reveal disease in the complaint line.\n"
    )
    client = AsyncOpenAI(api_key=openai_api_key)
    resp = await client.chat.completions.create(
        model=SCENARIO_MODEL,
        messages=[{"role": "user", "content": scenario_prompt}],
        max_tokens=100,
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    # parse out lines
    lines = text.split("\n")
    disease = "unknown"
    complaint = "I feel sick"
    for line in lines:
        if line.lower().startswith("disease:"):
            disease = line.split(":",1)[1].strip()
        elif line.lower().startswith("complaint:"):
            complaint = line.split(":",1)[1].strip()

    # We'll store them so our local model can embed them in the system prompt
    # e.g., "You are a doc+patient, the hidden disease is: X. The patient says: Y. Produce up to 5 doc/patient lines..."
    # Then the local model must produce "Final diagnosis: X" at the end.
    # (Yes, it's single-block self-play.)
    scenario = {
        "hidden_disease": disease,
        "complaint": complaint
    }
    return scenario

def build_prompt_for_local_model(disease: str, complaint: str, max_turns: int) -> str:
    """
    The text that the local model sees for a single scenario.
    The model should produce both sides of the conversation in one go, 
    then end with 'Final diagnosis: ...'
    """
    return f"""System:
You will produce a simulated conversation between 
(A) a patient with a hidden disease: {disease}, 
and (B) you, the AI Doctor.

You must produce up to {max_turns} pairs of lines (Doctor asks, Patient replies).
At the end, you must say "Final diagnosis: <disease>" 
matching the actual hidden disease: {disease}.

But do NOT reveal the hidden disease before the final line. 
The patient part should reflect consistent symptoms for {disease}.

Begin. The patient's first complaint is:
"{complaint}"
"""

###############################################################################
#                           REWARD FUNCTION
###############################################################################
async def judge_conversation(conversation: str, correct_disease: str, openai_api_key: str) -> float:
    """
    Calls GPT-4o-mini to judge:
    1) If it concluded with "Final diagnosis: correct_disease" => good
    2) If the conversation is coherent, etc.
    Returns a float in [0..1].
    We'll parse out the first float from GPT's response.
    """
    # We'll feed the entire conversation to GPT-4o-mini with instructions to rate it
    prompt = f"""System:
You are a judge for a medical conversation. 
You see the entire conversation below. The hidden disease was: {correct_disease}.

Rate from 0..1 how well this conversation:
1) Provided coherent question/answer pairs,
2) Ended with correct final diagnosis,
3) Did not reveal the disease prematurely.

Give the numeric score as the FIRST float in your reply. Explanation after if you like.

Conversation:
{conversation}
"""
    client = AsyncOpenAI(api_key=openai_api_key)
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )
    text = resp.choices[0].message.content
    match = re.search(r"\b(\d*\.?\d+)\b", text)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except:
            pass
    return 0.0

###############################################################################
#          MAIN SCRIPT: GATHER SCENARIOS, BUILD DATASET, RUN TRAIN
###############################################################################
async def main(openai_api_key: str):
    # 1) Load local doctor model
    doctor_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LOCAL_DOCTOR_MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
    )
    # 2) Create a standard GRPO LoRA
    doctor_model = FastLanguageModel.get_peft_model(
        doctor_model,
        r=8,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
        bias="none",
    )

    # 3) Set up a GRPOTrainer in the standard “Approach A” style
    config = GRPOConfig(
        use_vllm=True,
        learning_rate=LR,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=MAX_GLOBAL_STEPS,  
        max_grad_norm=0.3,
        num_generations=2,   # how many completions per prompt
        max_prompt_length=1024,
        max_completion_length=1024,
        save_steps=50,
        report_to="none",
        output_dir="grpo_outputs",
    )

    # We'll define a custom reward function that calls the judge (GPT-4o-mini).
    # The trainer automatically calls this function for each generation. 
    # We'll need to do it asynchronously, so we do a little trick:
    # - We'll store an event loop or we can do a synchronous wrapper.
    # But let's keep it simple: we do a "synchronous" call with 'asyncio.run(...)' inside the function,
    # which might not be best practice, but it's an example.
    async def _judge_batch(prompts, completions, diseases):
        # We'll do them one by one for clarity
        scores = []
        for prompt, completion_list, disease in zip(prompts, completions, diseases):
            # 'completion_list' is a list of 1 or more completions
            # We'll just pick the single completion if there's 1, or average them if there are multiple
            local_scores = []
            for c in completion_list:
                text = c["content"]  # a single big string with the entire conversation
                # call the judge
                sc = await judge_conversation(text, disease, openai_api_key)
                local_scores.append(sc)
            # Return the average for this prompt
            scores.append(sum(local_scores)/len(local_scores))
        return scores

    def sync_judge_func(prompts, completions, disease, **kwargs):
        # Wrap the async call
        return asyncio.run(_judge_batch(prompts, completions, disease))

    # We'll store a single function in reward_funcs:
    # But we must pass 'disease' from the dataset. The trainer automatically passes any dataset fields
    # that aren't named "prompt"/"completion" to the reward func as **kwargs.
    reward_funcs = [sync_judge_func]

    trainer = GRPOTrainer(
        model=doctor_model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=None,  # We'll set it below
        eval_dataset=None,
    )

    # We'll do multiple outer loops:
    for step_idx in range(NUM_OUTER_STEPS):
        logger.info(f"\n=== OUTER STEP {step_idx+1}/{NUM_OUTER_STEPS} ===")
        # (A) Generate 10 scenario seeds from GPT-4o-mini
        logger.info("Generating new scenario seeds...")
        scenarios = []
        for _ in range(SCENARIOS_PER_LOOP):
            scenario = await generate_scenario_seed(openai_api_key)
            scenarios.append(scenario)

        # (B) Build an HF Dataset from these scenarios
        # We'll create "prompt" text that instructs the local model to produce the entire conversation
        data_rows = []
        for sc in scenarios:
            disease = sc["hidden_disease"]
            complaint = sc["complaint"]
            full_prompt = build_prompt_for_local_model(disease, complaint, MAX_TURNS)
            # This becomes the "prompt." We'll also store "disease" so the reward_func sees it.
            data_rows.append({
                "prompt": [
                    {"role": "system", "content": "You are a medical diagnostic assistant."},
                    {"role": "user", "content": full_prompt}  # Changed from system to user role
                ],
                "disease": disease  # used by the reward func
            })
        ds = Dataset.from_list(data_rows)

        # (C) Set trainer's dataset
        trainer.train_dataset = ds
        # (D) Train
        # Because .train() calls the entire single-turn generation loop internally,
        # it will produce completions (the entire multi-turn conversation),
        # call sync_judge_func() -> judge_conversation(),
        # compute advantage, do updates, etc.
        logger.info("Beginning GRPO training on these new scenarios...")
        trainer.train()

        # (E) Save a checkpoint
        save_path = f"doctor_lora_outerstep_{step_idx+1}"
        logger.info(f"Saving checkpoint to {save_path}")
        doctor_model.save_pretrained(save_path)

    # Done
    logger.info("All steps complete. Saving final LoRA to doctor_lora_final")
    doctor_model.save_pretrained("doctor_lora_final")
    logger.info("Finished.")

###############################################################################
#                           SCRIPT ENTRY
###############################################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key")
    args = parser.parse_args()

    key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Please provide --openai_api_key or set OPENAI_API_KEY.")
    
    asyncio.run(main(key))
