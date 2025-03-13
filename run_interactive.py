"""
Interactive script for GRPODx Medical Diagnosis Agent.

This script loads a trained model and runs an interactive diagnostic session.
"""

import argparse
import sys
from unsloth import FastLanguageModel
from GRPODx_implementation import interactive_diagnosis
from config import MODEL_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Run GRPODx in interactive mode")
    parser.add_argument("--simulate", action="store_true", 
                      help="Run in simulation mode with a virtual patient")
    parser.add_argument("--model", type=str, default=MODEL_CONFIG["model_name"],
                      help=f"Model to use (default: {MODEL_CONFIG['model_name']})")
    args = parser.parse_args()
    
    print("Loading GRPODx Medical Diagnosis Agent...")
    
    # Load model parameters (matching training parameters)
    max_seq_length = MODEL_CONFIG["max_seq_length"]
    lora_rank = MODEL_CONFIG.get("lora_rank", 8)
    
    # Try to load model with fallback options
    model_options = MODEL_CONFIG.get("model_options", [args.model])
    if args.model and args.model not in model_options:
        model_options.insert(0, args.model)  # Try user-specified model first
    
    for model_name in model_options:
        try:
            print(f"Attempting to load model: {model_name}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=MODEL_CONFIG["load_in_4bit"],
                fast_inference=MODEL_CONFIG["fast_inference"],
                max_lora_rank=lora_rank,
            )
            print(f"Successfully loaded: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            continue
    else:
        print("ERROR: Could not load any of the available models.")
        print("Please check your environment or specify a valid model with --model")
        sys.exit(1)
        
    # Load LoRA weights if they exist
    print("Checking for trained LoRA weights (grpodx_model)...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=MODEL_CONFIG.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            lora_alpha=MODEL_CONFIG.get("lora_alpha", lora_rank),
        )
        model.load_adapter("grpodx_model", adapter_name="default")
        print("Successfully loaded fine-tuned weights!")
    except Exception as e:
        print(f"Note: Could not load fine-tuned weights: {e}")
        print("Running with base model only. Diagnoses may be less accurate.")
    
    print("Starting interactive diagnostic session...")
    print("-------------------------------------------")
    
    if args.simulate:
        print("SIMULATION MODE: A virtual patient will be generated with a random disease")
        print("-------------------------------------------\n")
        interactive_diagnosis(model, tokenizer, simulation_mode=True)
    else:
        print("Describe your symptoms and the AI doctor will ask you questions.")
        print("Type 'exit' to end the session.")
        print("-------------------------------------------\n")
        interactive_diagnosis(model, tokenizer, simulation_mode=False)

if __name__ == "__main__":
    main()