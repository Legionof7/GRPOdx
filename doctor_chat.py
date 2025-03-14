#!/usr/bin/env python3

import os
import re
import sys
import argparse
from typing import List, Dict, Tuple, Any

from unsloth import FastLanguageModel
from vllm import SamplingParams

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
        print(f"Loaded LoRA weights from {model_path}")
    except Exception as e:
        print(f"Error loading LoRA weights: {str(e)}")
        print("Falling back to using fresh model without LoRA weights")
        # Apply fresh LoRA config as fallback
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Using default rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        )
    
    print("\n=== Model loaded successfully! ===")
    print("\nAI Doctor Chat Session")
    print("Type 'exit' to end the conversation")
    print("Type 'debug' to toggle showing the reasoning")
    print("Type 'clear' to start a new conversation")
    print("-----------------------------------")
    
    show_reasoning = False
    conversation_history = []
    
    while True:
        user_input = input("\nPatient: ").strip()
        
        # Handle special commands
        if user_input.lower() == 'exit':
            print("\nEnding session. Goodbye!")
            break
        elif user_input.lower() == 'debug':
            show_reasoning = not show_reasoning
            print(f"\nDebug mode {'enabled' if show_reasoning else 'disabled'}")
            continue
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("\nConversation history cleared. Starting new conversation.")
            continue
        
        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate doctor's response
        full_response, visible_response = generate_doctor_turn(
            model, tokenizer, conversation_history, temperature
        )
        
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
    args = parser.parse_args()
    
    interactive_chat_session(args.model_path, args.temperature)

if __name__ == "__main__":
    main()