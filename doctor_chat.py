#!/usr/bin/env python3

import os
import re
import sys
import argparse
import datetime
import logging
import uuid
from typing import List, Dict, Tuple, Any

from unsloth import FastLanguageModel
from vllm import SamplingParams

# Set up logging configuration
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/chat_conversations", exist_ok=True)
log_filename = f"logs/doctor_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants from medical_grpo.py
DOCTOR_SYSTEM_INSTRUCTIONS = """System:
You are an AI Doctor. Each time you speak, you MUST include a hidden chain-of-thought
within <reason>...</reason> tags. 

After writing your hidden reasoning in <reason>, produce a short statement to the patient.

NEVER reveal the text inside <reason> to the patient. 
"""

def remove_reason_tags(text: str) -> str:
    """
    Strip <reason>...</reason> from Doctor's output to produce the text
    visible to the patient.
    """
    return re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL).strip()

def generate_doctor_turn(
    doctor_model,
    tokenizer,
    conversation_history: List[Dict[str, str]],
    temperature=0.7
) -> Tuple[str, str]:
    """
    Produces a single Doctor turn from the local model, returning:
      - (full_text_with_reason, visible_text_no_reason).
    """
    # Build a system prompt instructing the Doctor to include <reason>...
    system_message = {"role": "system", "content": DOCTOR_SYSTEM_INSTRUCTIONS}
    
    # Format messages for chat template
    messages = [system_message] + conversation_history
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=400,
        stop=["<|end_of_document|>", "<|endoftext|>"]
    )
    
    outputs = doctor_model.fast_generate(
        [prompt],
        sampling_params=sampling_params
    )
    
    if not outputs or not outputs[0].outputs:
        return ("<reason>Failed generation</reason> Sorry, I couldn't respond.", "Sorry, I couldn't respond.")
    
    full_text = outputs[0].outputs[0].text.strip()
    visible_text = remove_reason_tags(full_text)
    
    return full_text, visible_text

def interactive_chat_session(model_path: str, temperature: float = 0.7):
    """
    Run an interactive doctor-patient chat session with the specified model.
    """
    logger.info("\n=== Loading Medical AI Doctor Model ===")
    logger.info(f"Using model path: {model_path}")
    logger.info(f"Temperature: {temperature}")
    
    print("\n=== Loading Medical AI Doctor Model ===")
    print(f"Using model path: {model_path}")
    print(f"Temperature: {temperature}")
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-4",
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
    )
    
    # Load LoRA weights - first convert to regular model
    try:
        # Try loading LoRA weights directly
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        logger.info(f"Loaded LoRA weights from {model_path}")
        print(f"Loaded LoRA weights from {model_path}")
    except Exception as e:
        error_msg = f"Error loading LoRA weights: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        print("Falling back to using fresh model without LoRA weights")
        logger.info("Falling back to using fresh model without LoRA weights")
        # Apply fresh LoRA config as fallback
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Using default rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        )
    
    logger.info("Model loaded successfully!")
    print("\n=== Model loaded successfully! ===")
    print("\nAI Doctor Chat Session")
    print("Type 'exit' to end the conversation")
    print("Type 'debug' to toggle showing the reasoning")
    print("Type 'clear' to start a new conversation")
    print("-----------------------------------")
    
    show_reasoning = False
    conversation_history = []
    conversation_id = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting new chat session with ID: {conversation_id}")
    
    while True:
        user_input = input("\nPatient: ").strip()
        
        # Handle special commands
        if user_input.lower() == 'exit':
            logger.info(f"Chat session {conversation_id} ended by user")
            print("\nEnding session. Goodbye!")
            # Save conversation history to file before exiting
            save_conversation_to_file(conversation_history, conversation_id)
            break
        elif user_input.lower() == 'debug':
            show_reasoning = not show_reasoning
            print(f"\nDebug mode {'enabled' if show_reasoning else 'disabled'}")
            continue
        elif user_input.lower() == 'clear':
            # Save previous conversation if it's not empty
            if conversation_history:
                save_conversation_to_file(conversation_history, conversation_id)
            
            # Create a new conversation ID
            conversation_history = []
            conversation_id = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Starting new chat session with ID: {conversation_id}")
            print("\nConversation history cleared. Starting new conversation.")
            continue
        
        # Log user input
        logger.debug(f"Patient ({conversation_id}): {user_input}")
        
        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate doctor's response
        full_response, visible_response = generate_doctor_turn(
            model, tokenizer, conversation_history, temperature
        )
        
        # Log doctor's response
        logger.debug(f"Doctor ({conversation_id}): {visible_response}")
        if "<reason>" in full_response:
            reasoning_match = re.search(r"<reason>(.*?)</reason>", full_response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                logger.debug(f"Doctor reasoning ({conversation_id}): {reasoning}")
        
        # Add doctor's response to conversation history (with reasoning)
        conversation_history.append({"role": "assistant", "content": full_response})
        
        # Display response to user
        print(f"\nDoctor: {visible_response}")
        
        # If debug mode is enabled, show the reasoning
        if show_reasoning:
            reasoning_match = re.search(r"<reason>(.*?)</reason>", full_response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                print(f"\n[REASONING]: {reasoning}")

def save_conversation_to_file(conversation_history, conversation_id):
    """
    Save the complete conversation to a file.
    """
    if not conversation_history:
        return
        
    filename = f"logs/chat_conversations/{conversation_id}.txt"
    logger.info(f"Saving conversation {conversation_id} to {filename}")
    
    with open(filename, "w") as f:
        f.write(f"Chat ID: {conversation_id}\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=== CONVERSATION HISTORY ===\n\n")
        
        for message in conversation_history:
            role = "PATIENT" if message["role"] == "user" else "DOCTOR"
            content = message["content"]
            f.write(f"{role}:\n{content}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Medical AI Doctor Chat Interface")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="doctor_lora_final",
        help="Path to the trained LoRA model (default: doctor_lora_final)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()
    
    # Set log level based on command line argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    
    logger.info(f"Starting doctor_chat.py with log level: {args.log_level}")
    
    try:
        interactive_chat_session(args.model_path, args.temperature)
    except Exception as e:
        logger.error(f"Error in chat session: {str(e)}", exc_info=True)
        print(f"\nAn error occurred: {str(e)}")
        print("Check the log file for more details.")

if __name__ == "__main__":
    main()