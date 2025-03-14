"""
Medical Model Self-Play training with GRPO and Phi-4
====================================================

This script implements a doctor-patient self-play system for medical diagnosis
training using GRPO (Generative Reward Policy Optimization) with Unsloth's
optimized Phi-4 model.

The system works by:
1. Doctor model attempts to diagnose a hidden disease through conversation
2. Patient model (GPT-4o-mini) simulates a patient with a hidden disease
3. Reward model evaluates the doctor's performance
4. GRPO updates doctor model weights based on performance

For more information, see techspec.txt
"""

import re
import os
import json
import torch
import random
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass

# Unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported
# Import GRPOTrainer from unsloth for compatibility with unsloth's implementation
from unsloth import GRPOTrainer
# Import GRPOConfig from trl
from trl import GRPOConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_TURNS = 5
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
NUM_GENERATIONS = 4
LEARNING_RATE = 5e-6
MAX_STEPS = 1000
SAVE_STEPS = 100
BATCH_SIZE = 2

# Patient system prompt template
PATIENT_SYSTEM_PROMPT = """System:
You are a patient simulator for medical training.

1. Pick a common disease (choose one that most people would be familiar with).
2. Roleplay as if you have that disease, providing realistic symptoms, history, and concerns.
3. DO NOT reveal the disease unless the Doctor makes a final diagnosis or explicitly asks.
4. Answer questions truthfully based on the chosen disease's typical presentation.
5. Be realistic in your responses - show appropriate concern, confusion, or relief.
6. Your chosen disease is YOUR DECISION - don't make it too obscure.

Your disease choice: <decide on a disease now and roleplay consistently>
"""

# Doctor system prompt template
DOCTOR_SYSTEM_PROMPT = """System:
You are an AI Doctor trying to diagnose a patient's condition.

IMPORTANT: 
1. You MUST always include hidden reasoning in <reason>...</reason> tags.
2. After your reasoning, provide a concise patient-friendly response.
3. Never reveal your reasoning in <reason> tags to the patient.
4. You have a maximum of {max_turns} questions/exchanges.
5. On your final turn, you MUST provide "Final diagnosis: <your diagnosis>" if you haven't already.

Current turn: {current_turn}/{max_turns}

Conversation so far:
{conversation}
"""

# Reward model system prompt
REWARD_SYSTEM_PROMPT = """System:
You are a medical educator evaluating a doctor's diagnostic performance.

The patient had: {revealed_disease}

Score the doctor's performance on a scale of 0.0 to 1.0 based on:
1. Thoroughness of reasoning (<reason> blocks)
2. Relevance of questions asked
3. Accuracy of the final diagnosis compared to {revealed_disease}
4. Efficiency (reaching correct diagnosis in fewer turns is better)

Provide partial credit for diagnoses that are partially correct or in the right category.

First, analyze the conversation. Then provide a single score between 0.0 and 1.0 at the beginning of your response.

Example scoring:
- 0.9-1.0: Perfect or near-perfect diagnosis with excellent reasoning
- 0.7-0.8: Correct diagnosis with good reasoning
- 0.5-0.6: Partially correct diagnosis or good reasoning but incorrect conclusion
- 0.3-0.4: On the right track but missed key elements
- 0.0-0.2: Incorrect diagnosis with poor reasoning

Conversation to evaluate:
{conversation}
"""

def load_doctor_model(model_path: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Load the doctor model with Unsloth optimizations.
    
    Args:
        model_path: Optional path to a saved model/adapter
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading doctor model with Unsloth...")
    
    # Load base model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-4",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.8,
    )
    
    # Apply LoRA for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["gate_proj", "up_proj", "down_proj", 
                        "q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # If a saved model path is provided, load it
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading saved model from {model_path}")
        model.load_adapter(model_path)
    
    return model, tokenizer


def remove_reason_tags(text: str) -> str:
    """
    Remove <reason>...</reason> blocks from the text.
    
    Args:
        text: Text containing <reason> tags
        
    Returns:
        Text with <reason> blocks removed
    """
    return re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL).strip()


def format_conversation(conversation: List[Dict[str, str]], include_reason: bool = True) -> str:
    """
    Format a conversation into a string for prompting.
    
    Args:
        conversation: List of conversation turns
        include_reason: Whether to include <reason> blocks
        
    Returns:
        Formatted conversation string
    """
    formatted = []
    for turn in conversation:
        role = turn["role"].title()
        content = turn["content"]
        
        if role == "Doctor" and not include_reason:
            content = remove_reason_tags(content)
            
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)


async def get_patient_response(conversation: List[Dict[str, str]], 
                              api_key: str) -> Tuple[str, Optional[str]]:
    """
    Get a response from the patient model (GPT-4o-mini).
    
    Args:
        conversation: Conversation history without <reason> blocks
        api_key: OpenAI API key
        
    Returns:
        Tuple of (patient_response, revealed_disease if final turn else None)
    """
    import openai
    
    client = openai.AsyncOpenAI(api_key=api_key)
    
    # Format conversation for the patient model
    # We map doctor -> user, patient -> assistant
    # This is because from the patient model's perspective, it responds as the patient (assistant)
    # and receives messages from the doctor (user)
    messages = [{"role": "system", "content": PATIENT_SYSTEM_PROMPT}]
    
    for turn in conversation:
        role = "user" if turn["role"] == "doctor" else "assistant"
        content = turn["content"]
        if role == "user":
            content = remove_reason_tags(content)
        messages.append({"role": role, "content": content})
    
    # Check if this is the final turn (doctor made a diagnosis)
    is_final = any("final diagnosis:" in turn["content"].lower() 
                  for turn in conversation if turn["role"] == "doctor")
    
    # Get patient response
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )
    
    patient_response = response.choices[0].message.content
    
    # If final turn, get the revealed disease with a separate call
    revealed_disease = None
    if is_final:
        reveal_messages = messages.copy()
        reveal_messages.append({"role": "user", 
                              "content": "The doctor has made a final diagnosis. Please reveal what disease you were simulating."})
        
        reveal_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=reveal_messages,
            temperature=0.3,
            max_tokens=64
        )
        
        revealed_disease = reveal_response.choices[0].message.content
    
    return patient_response, revealed_disease


async def get_reward_score(conversation: List[Dict[str, str]], 
                          revealed_disease: str,
                          api_key: str) -> float:
    """
    Get a reward score for the doctor's performance from the reward model.
    
    Args:
        conversation: Full conversation with <reason> blocks
        revealed_disease: The disease revealed by the patient
        api_key: OpenAI API key
        
    Returns:
        Reward score between 0 and 1
    """
    import openai
    
    client = openai.AsyncOpenAI(api_key=api_key)
    
    # Format the conversation with reason blocks included
    formatted_conversation = format_conversation(conversation, include_reason=True)
    
    # Create prompt for the reward model
    reward_prompt = REWARD_SYSTEM_PROMPT.format(
        revealed_disease=revealed_disease,
        conversation=formatted_conversation
    )
    
    # Get reward score
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reward_prompt}],
        temperature=0.2,
        max_tokens=256
    )
    
    reward_text = response.choices[0].message.content
    
    # Extract score - find the first float between 0 and 1
    match = re.search(r"([0-9]*\.[0-9]+|[0-9]+)", reward_text)
    if match:
        try:
            score = float(match.group(0))
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
        except ValueError:
            logger.warning(f"Could not parse reward score from: {reward_text}")
    
    logger.warning("No valid score found, defaulting to 0.5")
    return 0.5


def generate_doctor_response(model, tokenizer, conversation, current_turn, temperature=0.7):
    """
    Generate a response from the doctor model.
    
    Args:
        model: The doctor model
        tokenizer: The tokenizer
        conversation: List of conversation turns
        current_turn: Current turn number
        temperature: Sampling temperature
        
    Returns:
        Doctor's response text
    """
    # Format conversation for the doctor
    formatted_conversation = format_conversation(conversation, include_reason=False)
    
    # Create system prompt
    system_prompt = DOCTOR_SYSTEM_PROMPT.format(
        max_turns=MAX_TURNS,
        current_turn=current_turn,
        conversation=formatted_conversation
    )
    
    # Create chat prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response with Unsloth's optimized inference
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=768,
    )
    
    response = model.fast_generate(
        [prompt],
        sampling_params=sampling_params,
    )[0]
    
    # Extract the generated text
    return response.outputs[0].text.strip()


async def run_conversation_episode(doctor_model, tokenizer, openai_api_key: str) -> Tuple[List[Dict[str, str]], float]:
    """
    Run a complete conversation episode between doctor and patient.
    
    Args:
        doctor_model: The doctor model
        tokenizer: The tokenizer
        openai_api_key: OpenAI API key
        
    Returns:
        Tuple of (conversation history, reward score)
    """
    conversation = []
    revealed_disease = None
    
    # First patient message (presenting complaint)
    first_patient_response, _ = await get_patient_response(
        [{"role": "system", "content": "The patient will now describe their main symptom or concern."}],
        openai_api_key
    )
    
    conversation.append({
        "role": "patient",
        "content": first_patient_response
    })
    
    # Run the conversation for MAX_TURNS or until a final diagnosis
    for turn in range(1, MAX_TURNS + 1):
        # Doctor turn
        doctor_response = generate_doctor_response(
            doctor_model, tokenizer, conversation, turn
        )
        
        conversation.append({
            "role": "doctor",
            "content": doctor_response
        })
        
        # Check if final diagnosis was made
        if "final diagnosis:" in doctor_response.lower():
            # Get revealed disease from patient
            _, revealed_disease = await get_patient_response(
                conversation, openai_api_key
            )
            break
        
        # Patient turn (if not the final turn)
        if turn < MAX_TURNS:
            patient_response, _ = await get_patient_response(
                conversation, openai_api_key
            )
            
            conversation.append({
                "role": "patient",
                "content": patient_response
            })
    
    # If we reached MAX_TURNS without a diagnosis, force final turn and get revealed disease
    if not revealed_disease:
        if "final diagnosis:" not in conversation[-1]["content"].lower():
            # Force final doctor turn with diagnosis
            current_conversation = format_conversation(conversation, include_reason=False)
            system_prompt = DOCTOR_SYSTEM_PROMPT.format(
                max_turns=MAX_TURNS,
                current_turn=MAX_TURNS,
                conversation=current_conversation
            ) + "\n\nThis is your final turn. You MUST provide a final diagnosis now."
            
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate final response with diagnosis
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=768,
            )
            
            final_response = doctor_model.fast_generate(
                [prompt],
                sampling_params=sampling_params,
            )[0].outputs[0].text.strip()
            
            # Replace or add the final doctor turn
            if conversation[-1]["role"] == "doctor":
                conversation[-1]["content"] = final_response
            else:
                conversation.append({
                    "role": "doctor",
                    "content": final_response
                })
        
        # Get revealed disease
        _, revealed_disease = await get_patient_response(
            conversation, openai_api_key
        )
    
    # Calculate reward
    reward = await get_reward_score(
        conversation, revealed_disease, openai_api_key
    )
    
    logger.info(f"Episode completed. Revealed disease: {revealed_disease}, Reward: {reward}")
    
    return conversation, reward


async def generate_batch_data(doctor_model, tokenizer, openai_api_key, batch_size=2, completions_per_scenario=4):
    """
    Generate batch data for GRPO training.
    
    Args:
        doctor_model: The doctor model
        tokenizer: The tokenizer
        openai_api_key: OpenAI API key
        batch_size: Number of scenarios per batch
        completions_per_scenario: Number of completions per scenario
        
    Returns:
        List of (conversation, advantage) pairs for GRPO
    """
    batch_data = []
    
    for _ in range(batch_size):
        scenario_completions = []
        
        # Run multiple completions for the same scenario
        for _ in range(completions_per_scenario):
            conversation, reward = await run_conversation_episode(
                doctor_model, tokenizer, openai_api_key
            )
            scenario_completions.append((conversation, reward))
        
        # Calculate advantages based on average reward for this scenario
        rewards = [completion[1] for completion in scenario_completions]
        avg_reward = sum(rewards) / len(rewards)
        advantages = [reward - avg_reward for reward in rewards]
        
        # Format data for GRPO trainer
        for (conversation, _), advantage in zip(scenario_completions, advantages):
            # Format into a single text for GRPO
            formatted_text = format_conversation(conversation, include_reason=True)
            batch_data.append((formatted_text, advantage))
    
    return batch_data


def setup_grpo_trainer(doctor_model, tokenizer):
    """
    Set up the GRPO trainer.
    
    Args:
        doctor_model: The doctor model
        tokenizer: The tokenizer
        
    Returns:
        GRPO trainer instance
    """
    # Define reward functions
    def conversation_quality_reward(completions, **kwargs) -> list[float]:
        """Reward function for conversation quality and format."""
        responses = [completion[0]['content'] for completion in completions]
        rewards = []
        
        for response in responses:
            reward = 0.0
            # Check for <reason> tags
            if "<reason>" in response and "</reason>" in response:
                reward += 0.5
            # Check for coherent response
            if len(response) > 50:
                reward += 0.3
            # Check for diagnostic language
            if any(term in response.lower() for term in ["diagnosis", "condition", "symptom", "treatment"]):
                reward += 0.2
            rewards.append(reward)
            
        return rewards
    
    # Configure GRPO
    # Note: Make sure all parameters are supported by your version of GRPOConfig
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=4,  # Ensure divisible by num_generations
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=1024,
        max_completion_length=1024,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs",
    )
    
    trainer = GRPOTrainer(
        model=doctor_model,
        processing_class=tokenizer,
        reward_funcs=[conversation_quality_reward],  # Add required reward functions
        args=training_args,
    )
    
    return trainer


async def main(openai_api_key):
    """Main training loop"""
    # Load model
    doctor_model, tokenizer = load_doctor_model()
    
    # Setup GRPO trainer
    trainer = setup_grpo_trainer(doctor_model, tokenizer)
    
    # Main training loop
    for step in range(MAX_STEPS):
        logger.info(f"Starting training step {step+1}/{MAX_STEPS}")
        
        # Generate batch data
        batch_data = await generate_batch_data(
            doctor_model, 
            tokenizer, 
            openai_api_key,
            batch_size=BATCH_SIZE,
            completions_per_scenario=NUM_GENERATIONS
        )
        
        # Train on batch
        trainer.train_on_records(batch_data)
        
        # Save checkpoint periodically
        if (step + 1) % SAVE_STEPS == 0:
            checkpoint_path = f"doctor_checkpoint_step_{step+1}"
            doctor_model.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    doctor_model.save_pretrained("doctor_final_model")
    logger.info("Training completed. Final model saved.")


if __name__ == "__main__":
    import asyncio
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Train a medical diagnosis model with GRPO")
    parser.add_argument("--openai-api-key", help="OpenAI API key for patient simulation (defaults to OPENAI_API_KEY env variable)")
    parser.add_argument("--load-checkpoint", help="Path to load checkpoint from")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Maximum training steps")
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS, help="Steps between checkpoints")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK, help="LoRA rank")
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided either via --openai-api-key argument or OPENAI_API_KEY environment variable")
    
    # Update global constants
    MAX_STEPS = args.max_steps
    SAVE_STEPS = args.save_steps
    BATCH_SIZE = args.batch_size
    LORA_RANK = args.lora_rank
    
    # Run the async main function
    asyncio.run(main(api_key))