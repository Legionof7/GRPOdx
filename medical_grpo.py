#!/usr/bin/env python3

import os
import re
import math
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import argparse

from unsloth import FastLanguageModel, is_bfloat16_supported
from openai import AsyncOpenAI
from trl import GRPOTrainer, GRPOConfig
from vllm import SamplingParams

###############################################################################
#                          CONFIG & GLOBAL CONSTANTS
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MAX_TURNS = 5
NUM_STEPS = 2                # Outer steps (for demonstration)
SCENARIOS_PER_STEP = 1       # How many scenarios per step
COMPLETIONS_PER_SCENARIO = 2 # Re-run scenario with different random seeds, etc.

OPENAI_API_MODEL = "gpt-4o-mini"

PATIENT_SYSTEM_PROMPT = """System:
You are a patient simulator for medical training.
1. You have chosen a disease to simulate (hidden).
2. Provide consistent symptoms for that disease.
3. Do not reveal it unless asked.
Begin roleplaying as a patient now.
"""

DOCTOR_SYSTEM_PROMPT = """System:
You are an AI Doctor. Each turn, you MUST include a hidden chain-of-thought 
within <reason>...</reason> tags, then a short question or final statement.
NEVER reveal <reason> to the patient. 
"""

###############################################################################
#                         LOAD DOCTOR MODEL
###############################################################################
def load_doctor_model(
    model_name: str = "unsloth/Phi-4",
    lora_rank: int = 8
):
    logger.info(f"Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
    )
    logger.info("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=2*lora_rank,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
        bias="none",
    )
    return model, tokenizer

def remove_reason_tags(text: str) -> str:
    """Strip <reason>...</reason> from the Doctor's output."""
    return re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL).strip()

###############################################################################
#                      OPENAI HELPER: PATIENT & CRITIC
###############################################################################
async def get_patient_reply(
    conversation_no_reason: List[Dict[str, str]],
    openai_api_key: str,
) -> str:
    """
    Calls GPT-4o-mini to get the next patient turn.
    The conversation is provided with roles "doctor"/"patient" => mapped to Chat roles.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    messages = [{"role": "system", "content": PATIENT_SYSTEM_PROMPT}]
    for turn in conversation_no_reason:
        role = "user" if turn["role"] == "doctor" else "assistant"
        messages.append({"role": role, "content": turn["content"]})
    
    resp = await client.chat.completions.create(
        model=OPENAI_API_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

async def reveal_disease(
    conversation_no_reason: List[Dict[str, str]],
    openai_api_key: str
) -> str:
    """After final diagnosis, ask the patient model to reveal the hidden disease."""
    client = AsyncOpenAI(api_key=openai_api_key)
    messages = [{"role": "system", "content": PATIENT_SYSTEM_PROMPT}]
    for turn in conversation_no_reason:
        role = "user" if turn["role"] == "doctor" else "assistant"
        messages.append({"role": role, "content": turn["content"]})
    
    messages.append({
        "role": "user",
        "content": "The Doctor gave a final diagnosis. Please reveal the hidden disease now."
    })
    
    resp = await client.chat.completions.create(
        model=OPENAI_API_MODEL,
        messages=messages,
        max_tokens=50,
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()

async def get_turn_reward(
    conversation_no_reason: List[Dict[str, str]],
    doctor_turn: str,
    openai_api_key: str
) -> float:
    """
    Partial reward for the new Doctor turn's coherence & relevance.
    We'll prompt GPT-4o-mini with *just* the conversation so far + new doc turn,
    asking for a 0..0.5 score. 
    You can design your own scale or criteria.

    Returns a float in [0..0.5].
    """
    # Build a short system prompt that instructs the critic to rate only the last doc turn
    turn_critic_prompt = f"""System:
You are a turn-by-turn conversation critic. 
Look at the entire conversation so far (no hidden reasons), 
then the new Doctor turn. 
Rate how relevant & coherent the new Doctor turn is in [0..0.5]. 
Just provide a float as the first number.
Conversation so far:
"""
    conv_text = []
    for turn in conversation_no_reason:
        conv_text.append(f"{turn['role'].title()}: {turn['content']}")
    conv_text.append(f"Doctor (new turn): {doctor_turn}")
    critic_input = turn_critic_prompt + "\n".join(conv_text)
    
    client = AsyncOpenAI(api_key=openai_api_key)
    resp = await client.chat.completions.create(
        model=OPENAI_API_MODEL,
        messages=[{"role": "user", "content": critic_input}],
        max_tokens=100,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    match = re.search(r"(\d+(\.\d+)?)", raw)
    if match:
        val = float(match.group(1))
        # clamp to [0..0.5]
        return max(0.0, min(0.5, val))
    return 0.0

async def get_final_correctness_reward(
    conversation_with_reason: List[Dict[str, str]],
    revealed_disease: str,
    openai_api_key: str
) -> float:
    """
    Additional reward for final correctness.
    We'll show the entire conversation including <reason> blocks to a 'final diagnosis critic'
    that yields a float [0..0.5].
    """
    final_critic_prompt = f"""System:
You are a final diagnosis critic. 
See the entire conversation (with <reason> blocks). 
The actual disease was: {revealed_disease}.
If the Doctor's final diagnosis matches it, give up to 0.5. 
If partially correct, partial credit. 
If no final diagnosis or completely incorrect, 0.

Conversation:
"""
    conv_text = []
    for turn in conversation_with_reason:
        role = turn["role"].title()
        content = turn["content"]
        conv_text.append(f"{role}: {content}")
    full_text = "\n".join(conv_text)
    prompt_content = final_critic_prompt + full_text
    
    client = AsyncOpenAI(api_key=openai_api_key)
    resp = await client.chat.completions.create(
        model=OPENAI_API_MODEL,
        messages=[{"role": "user", "content": prompt_content}],
        max_tokens=150,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    match = re.search(r"(\d+(\.\d+)?)", raw)
    if match:
        val = float(match.group(1))
        return max(0.0, min(0.5, val))
    return 0.0

###############################################################################
#                 DOCTOR INFERENCE (LOCAL MODEL) FUNCTION
###############################################################################
def generate_doctor_turn(
    doctor_model,
    tokenizer,
    conversation_no_reason: List[Dict[str, str]],
    turn_idx: int,
    max_turns: int,
    temperature=0.9
) -> Tuple[str, str]:
    """
    Generate one Doctor turn from local model. 
    Returns (full_text_with_reason, visible_text_no_reason).
    """
    system_prompt = DOCTOR_SYSTEM_PROMPT
    
    # Build conversation so far
    conv_text = []
    for turn in conversation_no_reason:
        conv_text.append(f"{turn['role'].title()}: {turn['content']}")
    partial_history = "\n".join(conv_text)
    
    # We'll end with "Doctor:" so the model continues
    prompt_for_model = (
        f"{system_prompt}\n"
        f"Current turn: {turn_idx}/{max_turns}\n"
        f"Conversation so far:\n{partial_history}\n"
        f"Doctor:"
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=300,
        stop=["<|end_of_document|>", "<|endoftext|>"]
    )
    outputs = doctor_model.fast_generate(
        [prompt_for_model], sampling_params=sampling_params
    )
    if not outputs or not outputs[0].outputs:
        return ("<reason>Fail</reason> Sorry.", "Sorry.")
    
    full_doc = outputs[0].outputs[0].text.strip()
    visible_doc = remove_reason_tags(full_doc)
    return full_doc, visible_doc

###############################################################################
#            RUN ONE SCENARIO: MULTI-BLOCK RL (Approach B)
###############################################################################
async def run_selfplay_episode(
    doctor_model,
    tokenizer,
    trainer: GRPOTrainer,
    openai_api_key: str,
    max_turns: int,
    episode_id: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Runs a multi-turn scenario. 
    For each Doctor turn, we do:
      - partial reward for coherence
      - if final diagnosis is present, we do final correctness reward
      - sum them, do .grpo_step with (prompt, doc_turn)
    Returns:
      conversation_no_reason,
      conversation_with_reason
    """
    logger.info(f"[Episode #{episode_id}] Starting scenario with up to {max_turns} turns.")
    conversation_nr = []
    conversation_wr = []
    
    # Patient's first response
    system_kickoff = [{"role": "doctor", "content": "Hello, what's your main complaint?"}]
    first_patient = await get_patient_reply(system_kickoff, openai_api_key)
    conversation_nr.append({"role": "patient", "content": first_patient})
    conversation_wr.append({"role": "patient", "content": first_patient})
    
    for turn_idx in range(1, max_turns+1):
        # Doctor generates a new turn
        full_doc, visible_doc = generate_doctor_turn(
            doctor_model, tokenizer, conversation_nr, turn_idx, max_turns
        )
        conversation_wr.append({"role": "doctor", "content": full_doc})
        conversation_nr.append({"role": "doctor", "content": visible_doc})
        
        # 1) partial reward for coherence, etc.
        partial_reward = await get_turn_reward(conversation_nr[:-1], visible_doc, openai_api_key)
        # conversation_nr[:-1] is the conversation before this new doc turn
        # so the critic sees context + the doc's new turn = partial_reward in [0..0.5]
        
        # 2) final correctness reward if "Final diagnosis" found
        final_reward = 0.0
        if "final diagnosis:" in visible_doc.lower():
            # We end the scenario after this. 
            # Let's see if the doc is correct
            revealed = await reveal_disease(conversation_nr, openai_api_key)
            final_reward = await get_final_correctness_reward(conversation_wr, revealed, openai_api_key)
            logger.info(f"Doctor gave final diagnosis. Additional final reward={final_reward:.3f}")
        
        total_reward = partial_reward + final_reward
        
        # 3) RL update: we treat "conversation so far" as prompt, new doc turn as completion
        # We'll flatten the conversation up to but not including the new doc turn as prompt
        # Actually, TRL expects a "list of roles", so we can do:
        # prompts = the entire conversation minus this last doc turn
        # completions = just this doc turn
        # advantage = total_reward (minus baseline if you want)
        
        # Build the prompt for .grpo_step
        # We'll put everything so far in user role. 
        # In practice, you can do a simpler "User: Summarize the conversation so far..."
        # but let's do a direct approach:
        
        prompt_text_lines = []
        for t in conversation_nr[:-1]:
            prompt_text_lines.append(f"{t['role'].upper()}: {t['content']}")
        prompt_text = "\n".join(prompt_text_lines)
        
        # The new Doctor turn is the "completion"
        doc_completion = visible_doc
        
        # Optionally do baseline advantage = reward - baseline. 
        # For simplicity, we skip baseline => advantage = total_reward
        advantage = total_reward
        
        # Single batch => pass as a list of lists
        prompts = [[{"role": "user", "content": prompt_text}]]
        completions = [[{"role": "assistant", "content": doc_completion}]]
        
        stats = trainer.grpo_step(
            prompts=prompts,
            completions=completions,
            advantage=[advantage],
        )
        
        logger.info(f"[turn {turn_idx}] partial={partial_reward:.3f}, final={final_reward:.3f}, total={total_reward:.3f}, "
                    f"loss={stats['loss']:.4f}, kl={stats['kl']:.4f}, reward={stats['reward']:.4f}, "
                    f"grad_norm={stats['grad_norm']:.4f}")
        
        if final_reward > 0 or "final diagnosis:" in visible_doc.lower():
            logger.info(f"Episode #{episode_id} ended on turn {turn_idx}.")
            break
        
        # 4) If not final, get patient reply
        if turn_idx < max_turns:
            pat_resp = await get_patient_reply(conversation_nr, openai_api_key)
            conversation_nr.append({"role": "patient", "content": pat_resp})
            conversation_wr.append({"role": "patient", "content": pat_resp})
    
    return conversation_nr, conversation_wr

###############################################################################
#                     MAIN TRAINING LOOP
###############################################################################
async def main(openai_api_key: str):
    logger.info("=== Approach B: Multi-Block RL, partial turn rewards & final correctness ===")
    logger.info(f"MAX_TURNS={MAX_TURNS}, NUM_STEPS={NUM_STEPS}, SCENARIOS_PER_STEP={SCENARIOS_PER_STEP}, COMPLETIONS_PER_SCENARIO={COMPLETIONS_PER_SCENARIO}")
    
    doctor_model, tokenizer = load_doctor_model()
    
    # Configure GRPO
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
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
        max_steps=999999,  
        max_grad_norm=0.3,
        num_generations=1, 
        max_prompt_length=2048,
        max_completion_length=1024,
        save_steps=1,
        report_to="none",
        output_dir="doctor_multi_block_outputs",
    )
    
    # No dataset, no reward_funcs => manual .grpo_step
    trainer = GRPOTrainer(
        model=doctor_model,
        processing_class=tokenizer,
        reward_funcs=[],
        args=training_args,
    )
    
    episode_id = 0
    
    for step_idx in range(NUM_STEPS):
        logger.info(f"\n=== Outer Step {step_idx+1}/{NUM_STEPS} ===")
        
        for sc_idx in range(SCENARIOS_PER_STEP):
            for c_idx in range(COMPLETIONS_PER_SCENARIO):
                episode_id += 1
                logger.info(f"\n--- Scenario {sc_idx+1}, Completion {c_idx+1}, Episode #{episode_id} ---")
                
                # Run scenario with multi-turn partial shaping
                conversation_nr, conversation_wr = await run_selfplay_episode(
                    doctor_model,
                    tokenizer,
                    trainer,
                    openai_api_key,
                    MAX_TURNS,
                    episode_id
                )
        
        # Save checkpoint each outer step
        ckpt_path = f"doctor_lora_step{step_idx+1}"
        logger.info(f"Saving checkpoint to {ckpt_path} ...")
        doctor_model.save_pretrained(ckpt_path)
    
    # Final save
    logger.info("All steps done. Saving final LoRA to doctor_lora_final ...")
    doctor_model.save_pretrained("doctor_lora_final")
    logger.info("Done.")

###############################################################################
#                           ENTRY POINT
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-block RL: partial turn rewards + final correctness for multi-turn Doctor–Patient.")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key or from env OPENAI_API_KEY")
    args = parser.parse_args()
    
    key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Must provide --openai_api_key or set OPENAI_API_KEY env var.")
    
    asyncio.run(main(key))
