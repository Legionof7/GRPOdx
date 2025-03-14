#!/usr/bin/env python3

import os
import re
import math
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional

from unsloth import FastLanguageModel, is_bfloat16_supported
from openai import AsyncOpenAI
from trl import GRPOTrainer, GRPOConfig

###############################################################################
#                          CONFIG & GLOBAL CONSTANTS
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Adjust these as needed for your training
MAX_TURNS = 5       # Max Doctor–Patient exchanges
NUM_STEPS = 10      # How many outer loops of scenario generation & training
SCENARIOS_PER_STEP = 2        # How many different patient scenarios per step
COMPLETIONS_PER_SCENARIO = 2  # Number of completions to compute advantage
OPENAI_API_MODEL = "gpt-4o-mini"    # or "gpt-4o-mini" if available

# Prompts
PATIENT_SYSTEM_PROMPT = """System:
You are a patient simulator for medical training.

Steps:
1. You have chosen (or will choose now) a common disease to simulate. Keep it hidden.
2. Provide realistic symptoms and answers consistent with that disease.
3. Do not reveal it unless the Doctor explicitly makes a final diagnosis or asks for the reveal.
4. Answer truthfully for that condition. Remain consistent.

Begin roleplaying as a patient now.
"""

DOCTOR_SYSTEM_INSTRUCTIONS = """System:
You are an AI Doctor. Each time you speak, you MUST include a hidden chain-of-thought
within <reason>...</reason> tags. 

After writing your hidden reasoning in <reason>, produce a short statement to the patient.

NEVER reveal the text inside <reason> to the patient. 
You have a maximum of {max_turns} total question/answer exchanges.
By your final turn (turn {max_turns}), you MUST provide 
"Final diagnosis: <disease>" if you have not already.
"""

REWARD_SYSTEM_PROMPT = """System:
You are a medical conversation evaluator. You see the ENTIRE conversation below,
including the Doctor's hidden <reason> blocks. The patient was actually suffering from: {revealed_disease}.

Please provide a single floating-point score in [0..1] **as the first float** in your response,
based on:
1) How coherent/thorough the Doctor's <reason> blocks are,
2) Whether the final diagnosis matches {revealed_disease},
3) Quality and relevance of Doctor's questions.

If the diagnosis is partially correct, give partial credit. 
Write any explanation after the numeric score, but the first float you mention is the official score.

Conversation:
{conversation}
"""

###############################################################################
#                          MODEL LOADING & PREP
###############################################################################
def load_doctor_model(model_name: str = "unsloth/Phi-4",
                      lora_rank: int = 8) -> Tuple[Any, Any]:
    """
    Load the base Unsloth model + tokenizer, then apply LoRA.
    """
    logger.info(f"Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,      # 4-bit quant for efficiency
        fast_inference=True,
        gpu_memory_utilization=0.7,  # Reduced from 0.9 to avoid OOM errors
        enforce_eager=True,     # Use eager mode instead of cudagraphs to avoid OOM during capture
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
    """
    Strip <reason>...</reason> from Doctor's output to produce the text
    visible to the patient.
    """
    return re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL).strip()

###############################################################################
#                          OPENAI HELPER FUNCTIONS
###############################################################################
async def get_patient_reply(
    conversation_no_reason: List[Dict[str, str]],
    openai_api_key: str,
) -> str:
    """
    Calls GPT-4 (or GPT-4o-mini) to get the next Patient turn.
    conversation_no_reason should NOT contain <reason> blocks.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    
    # We map conversation roles to "user"/"assistant" for ChatCompletion
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
    """
    After the final Doctor turn, instruct the same Patient model
    to reveal the disease it was simulating.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    
    messages = [{"role": "system", "content": PATIENT_SYSTEM_PROMPT}]
    for turn in conversation_no_reason:
        role = "user" if turn["role"] == "doctor" else "assistant"
        messages.append({"role": role, "content": turn["content"]})
    
    # Now ask for the reveal
    reveal_msg = {
        "role": "user",
        "content": (
            "The Doctor made a final diagnosis. Please reveal which disease you were simulating."
        )
    }
    messages.append(reveal_msg)
    
    resp = await client.chat.completions.create(
        model=OPENAI_API_MODEL,
        messages=messages,
        max_tokens=100,
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()

async def get_conversation_reward(
    conversation_with_reason: List[Dict[str, str]],
    revealed_disease: str,
    openai_api_key: str
) -> float:
    """
    Calls GPT-4 (or GPT-4o-mini) to get a numeric reward in [0..1].
    We parse out the first float in its response.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    
    # Prepare conversation text with <reason> blocks included
    formatted_conv = []
    for turn in conversation_with_reason:
        role = turn["role"].title()
        content = turn["content"]
        formatted_conv.append(f"{role}: {content}")
    conv_str = "\n".join(formatted_conv)
    
    # Build the reward prompt
    prompt = REWARD_SYSTEM_PROMPT.format(
        revealed_disease=revealed_disease,
        conversation=conv_str
    )
    
    resp = await client.chat.completions.create(
        model=OPENAI_API_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,  # be more deterministic for scoring
    )
    reward_text = resp.choices[0].message.content
    
    # Extract the first float in [0..1]
    match = re.search(r"\b(\d*\.?\d+)\b", reward_text)
    if match:
        try:
            score = float(match.group(1))
            # clamp to [0..1]
            return max(0.0, min(1.0, score))
        except:
            pass
    # fallback
    return 0.0

###############################################################################
#                   DOCTOR INFERENCE (LOCAL MODEL) FUNCTION
###############################################################################
def generate_doctor_turn(
    doctor_model,
    tokenizer,
    conversation_no_reason: List[Dict[str, str]],
    turn_idx: int,
    max_turns: int,
    temperature=0.7
) -> Tuple[str, str]:
    """
    Produces a single Doctor turn from the local model, returning:
      - (full_text_with_reason, visible_text_no_reason).
    """
    # Build a system prompt instructing the Doctor to include <reason>...
    system_prompt = DOCTOR_SYSTEM_INSTRUCTIONS.format(
        max_turns=max_turns
    )
    
    # Format the conversation so far (excluding any <reason> blocks).
    # We interpret roles as "User" for patient, "Assistant" for doctor, etc.
    # But an easy approach is just to feed them in as strings:
    conversation_text = []
    for turn in conversation_no_reason:
        role = turn["role"].title()
        text = turn["content"]
        conversation_text.append(f"{role}: {text}")
    partial_history_str = "\n".join(conversation_text)
    
    full_system = (
        f"{system_prompt}\n\n"
        f"Current turn: {turn_idx}/{max_turns}\n\n"
        f"Conversation so far:\n{partial_history_str}\n"
        f"Doctor:"  # We want the model to continue from "Doctor:"
    )
    
    # Use the Unsloth fast_generate interface
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=300,
        stop=["<|end_of_document|>", "<|endoftext|>"]
    )
    
    outputs = doctor_model.fast_generate(
        [full_system],
        sampling_params=sampling_params
    )
    if not outputs or not outputs[0].outputs:
        return ("<reason>Failed generation</reason> Sorry, I couldn't respond.", "Sorry, I couldn't respond.")
    
    full_text = outputs[0].outputs[0].text.strip()
    
    # Visible text is the full text minus <reason> blocks
    visible_text = remove_reason_tags(full_text)
    
    return full_text, visible_text

###############################################################################
#                 RUN ONE SCENARIO (SELF-PLAY EPISODE)
###############################################################################
async def run_selfplay_episode(
    doctor_model,
    tokenizer,
    openai_api_key: str,
    max_turns: int = MAX_TURNS,
    episode_id: int = 0
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], float]:
    """
    Runs a single scenario to completion. Returns:
      - conversation_no_reason  (list of turns, no <reason>)
      - conversation_with_reason (list of turns, includes <reason> in Doctor turns)
      - final_reward (float)
    """
    logger.info(f"Starting episode #{episode_id}")
    conversation_no_reason = []
    conversation_with_reason = []
    
    # First, we ask the patient for an initial statement
    # We treat "doctor" as role for a system-level "kick-off"
    # or we can just call it "system: Provide your initial symptom."
    # But let's keep it simple:
    system_kickoff = [
        {"role": "doctor", "content": "Please describe your main symptom or complaint."}
    ]
    logger.info(f"Episode #{episode_id}: Requesting initial patient symptoms...")
    patient_reply = await get_patient_reply(system_kickoff, openai_api_key)
    logger.info(f"Episode #{episode_id}: Patient initial: {patient_reply}")
    conversation_no_reason.append({"role": "patient", "content": patient_reply})
    conversation_with_reason.append({"role": "patient", "content": patient_reply})
    
    # Conduct up to max_turns:
    for turn_idx in range(1, max_turns + 1):
        # Doctor turn
        logger.info(f"Episode #{episode_id}: Turn {turn_idx}/{max_turns} - Doctor thinking...")
        full_doc, doc_visible = generate_doctor_turn(
            doctor_model,
            tokenizer,
            conversation_no_reason,
            turn_idx,
            max_turns
        )
        logger.info(f"Episode #{episode_id}: Doctor visible response: {doc_visible}")
        conversation_with_reason.append({"role": "doctor", "content": full_doc})
        conversation_no_reason.append({"role": "doctor", "content": doc_visible})
        
        # Check if final diagnosis is in doc_visible:
        if "final diagnosis:" in doc_visible.lower():
            logger.info(f"Episode #{episode_id}: Final diagnosis detected, ending conversation")
            break
        
        # Patient turn (only if not final)
        if turn_idx < max_turns:
            logger.info(f"Episode #{episode_id}: Turn {turn_idx}/{max_turns} - Patient responding...")
            pat_resp = await get_patient_reply(conversation_no_reason, openai_api_key)
            logger.info(f"Episode #{episode_id}: Patient response: {pat_resp}")
            conversation_no_reason.append({"role": "patient", "content": pat_resp})
            conversation_with_reason.append({"role": "patient", "content": pat_resp})
    
    # Conversation ended (either due to final dx or hitting max_turns).
    # Reveal the hidden disease from the patient side
    logger.info(f"Episode #{episode_id}: Requesting disease reveal...")
    revealed = await reveal_disease(conversation_no_reason, openai_api_key)
    logger.info(f"Episode #{episode_id}: Disease revealed: {revealed}")
    
    # Now get the final reward from GPT-4 (or GPT-4o-mini) using the entire conversation_with_reason
    logger.info(f"Episode #{episode_id}: Computing conversation reward...")
    reward = await get_conversation_reward(conversation_with_reason, revealed, openai_api_key)
    
    logger.info(f"Episode #{episode_id} finished. Disease = {revealed}, Reward = {reward:.4f}")
    
    # Print complete conversation for analysis
    logger.info(f"Episode #{episode_id}: Complete conversation:")
    for i, turn in enumerate(conversation_with_reason):
        role = turn["role"].upper()
        logger.info(f"  [{i+1}] {role}: {turn['content'][:150]}{'...' if len(turn['content']) > 150 else ''}")
    
    return conversation_no_reason, conversation_with_reason, reward

###############################################################################
#                MAIN TRAINING LOOP: SCENARIO + ADVANTAGE + GRPO
###############################################################################
async def main(openai_api_key: str):
    """
    Main entry point:
    1. Load Doctor model & LoRA
    2. Create a GRPOTrainer
    3. Repeatedly generate new scenarios on-the-fly, compute advantage, update policy
    """
    logger.info("=== GRPO Doctor-Patient Medical Training ===")
    logger.info(f"Configuration: MAX_TURNS={MAX_TURNS}, NUM_STEPS={NUM_STEPS}")
    logger.info(f"SCENARIOS_PER_STEP={SCENARIOS_PER_STEP}, COMPLETIONS_PER_SCENARIO={COMPLETIONS_PER_SCENARIO}")
    logger.info(f"Using OpenAI API Model: {OPENAI_API_MODEL}")
    
    # 1. Load doctor model & tokenizer
    logger.info("Loading doctor model & tokenizer...")
    doctor_model, tokenizer = load_doctor_model()
    
    # 2. Configure GRPO
    logger.info("Configuring GRPO trainer...")
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,  # log every step
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=2,  # Increased to work with num_generations
        gradient_accumulation_steps=1,
        max_steps=999999,  # We'll do our own loop, so set large
        max_grad_norm=0.3,
        num_generations=2,         # Must be divisible by batch size
        max_prompt_length=1024,    # Adjust if needed
        max_completion_length=1024,
        save_steps=999999,         # We'll do custom saving
        report_to="none",
        output_dir="doctor_outputs",
    )
    # Create a custom dataset class from our batch data for the trainer
    class DoctorDataset:
        def __init__(self, conversations, advantages):
            self.conversations = conversations
            self.advantages = advantages
        
        def __len__(self):
            return len(self.conversations)
        
        def __getitem__(self, idx):
            return {
                "prompt": [{"role": "user", "content": self.conversations[idx]}],
                "advantage": self.advantages[idx]
            }
    
    # We'll build the dataset as we collect data, then train later
    all_text_examples = []
    all_advantages = []
    
    trainer = GRPOTrainer(
        model=doctor_model,
        processing_class=tokenizer,
        reward_funcs=[],  # We'll supply advantages directly
        args=training_args,
    )
    
    logger.info("Starting self-play GRPO training!")
    episode_counter = 0
    
    for step in range(NUM_STEPS):
        logger.info(f"\n=== Outer Step {step+1}/{NUM_STEPS} ===")
        
        # We'll gather data from SCENARIOS_PER_STEP scenarios,
        # each scenario we do COMPLETIONS_PER_SCENARIO completions
        # then compute advantage
        batch_data = []
        
        for sc_idx in range(SCENARIOS_PER_STEP):
            scenario_completions = []
            
            logger.info(f"Scenario {sc_idx+1}/{SCENARIOS_PER_STEP}")
            # We run the same "initial patient condition" multiple times
            # but strictly speaking you might get different initial patient answers
            # each time you call. In practice, you can fix the first patient
            # answer or let GPT pick a new disease each run.
            # We'll just let it pick new each time for diversity.
            
            for c_idx in range(COMPLETIONS_PER_SCENARIO):
                episode_counter += 1
                logger.info(f"=== Starting Episode #{episode_counter} (Step {step+1}, Scenario {sc_idx+1}, Completion {c_idx+1}) ===")
                
                conv_nr, conv_wr, reward = await run_selfplay_episode(
                    doctor_model,
                    tokenizer,
                    openai_api_key,
                    max_turns=MAX_TURNS,
                    episode_id=episode_counter
                )
                scenario_completions.append((conv_nr, conv_wr, reward))
            
            # Compute advantage
            rewards = [x[2] for x in scenario_completions]
            avg_reward = sum(rewards) / len(rewards)
            advs = [r - avg_reward for r in rewards]
            
            logger.info(f"Scenario {sc_idx+1} complete. Computing advantages...")
            logger.info(f"Rewards: {[f'{r:.4f}' for r in rewards]}")
            logger.info(f"Average reward: {avg_reward:.4f}")
            logger.info(f"Advantages: {[f'{a:.4f}' for a in advs]}")
            
            # Convert each completion into a training record
            # Typically you'd choose either:
            # (A) The entire conversation, or
            # (B) Just the final Doctor turn, depending on how your RL library expects it.
            #
            # Here, for simplicity, we feed the entire "visible" conversation as text,
            # and attach advantage. The GRPOTrainer will treat the entire text
            # as a single "input → output" chunk. This might not be strictly correct
            # for chat-based RL, but it’s a minimal example.
            
            for i, (conv_nr, conv_wr, rew) in enumerate(scenario_completions):
                # Flatten conversation_no_reason for training
                text_input = []
                for turn in conv_nr:
                    role = turn["role"].upper()
                    text_input.append(f"{role}: {turn['content']}")
                joined_text = "\n".join(text_input)
                
                advantage = advs[i]
                logger.info(
                    f" Completion {i+1}/{COMPLETIONS_PER_SCENARIO} reward={rew:.3f}, advantage={advantage:.3f}"
                )
                batch_data.append((joined_text, advantage))
        
        # Now we train on this batch_data
        logger.info(f"Training on batch of size {len(batch_data)} from step {step+1}")
        
        # Create a simple dataset from the batch data
        texts = [text for text, _ in batch_data]
        advantages = [adv for _, adv in batch_data]
        
        # Create a dataset object for training
        train_dataset = DoctorDataset(texts, advantages)
        
        # Update the trainer with the new dataset
        trainer.train_dataset = train_dataset
        
        # Train for 1 epoch on this small dataset
        # This avoids the need for grpo_step() and follows the notebook pattern
        try:
            trainer.train()
            logger.info(f"Completed training on batch from step {step+1}")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.info("Continuing despite training error")
        
        # Save checkpoint every second step
        if step % 2 == 1:
            logger.info(f"Saving checkpoint at step {step+1}...")
            save_path = f"doctor_lora_step{step+1}"
            doctor_model.save_pretrained(save_path)
            logger.info(f"Checkpoint saved to {save_path}")
    
    # Done. Save final LoRA
    logger.info("All training steps completed. Saving final model ...")
    doctor_model.save_pretrained("doctor_lora_final")
    logger.info("Done!")

###############################################################################
#                           SCRIPT ENTRY POINT
###############################################################################
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Doctor–Patient Self-Play with GRPO")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key (can also come from OPENAI_API_KEY env var)")
    args = parser.parse_args()
    
    key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Must provide --openai_api_key or set OPENAI_API_KEY env var.")
    
    # Run the async main
    asyncio.run(main(key))
