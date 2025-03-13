"""
Evaluation script for GRPODx Medical Diagnosis Agent.

This script loads a trained model and runs evaluation on test scenarios.
"""

import os
# Disable HF transfer to fix download issues
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from unsloth import FastLanguageModel
from test_scenarios import evaluate_model, TEST_DISEASE_BANK
import argparse
from config import MODEL_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPODx model")
    parser.add_argument("--num_cases", type=int, default=3, 
                      help="Number of test cases to evaluate")
    parser.add_argument("--test_all", action="store_true",
                      help="Test on all available disease scenarios")
    parser.add_argument("--use_verdict", action="store_true", default=True,
                      help="Use Verdict-based reward scoring with GPT-4o")
    parser.add_argument("--no_verdict", action="store_true",
                      help="Disable Verdict-based reward scoring")
    args = parser.parse_args()
    
    # --no_verdict overrides --use_verdict
    if args.no_verdict:
        args.use_verdict = False
    
    print("Loading GRPODx Medical Diagnosis Agent...")
    
    # Load model parameters from config
    max_seq_length = MODEL_CONFIG["max_seq_length"]
    lora_rank = MODEL_CONFIG.get("lora_rank", 8)
    
    # Try models in order until one works
    model_options = MODEL_CONFIG["model_options"]
    for model_name in model_options:
        try:
            print(f"Attempting to load model: {model_name}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=MODEL_CONFIG.get("load_in_4bit", True),
                fast_inference=MODEL_CONFIG.get("fast_inference", True),
                max_lora_rank=lora_rank,
                gpu_memory_utilization=MODEL_CONFIG.get("gpu_memory_utilization", 0.9),
            )
            print(f"Successfully loaded: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            continue
    else:
        raise ValueError("Could not load any of the available models. Please check your environment.")
    
    # Load LoRA weights
    print("Loading LoRA weights from grpodx_model...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_rank,
        )
        model.load_adapter("grpodx_model", adapter_name="default")
        print("Successfully loaded LoRA weights")
    except Exception as e:
        print(f"Warning: Failed to load LoRA weights: {e}")
        print("Continuing with base model only")
    
    # Display reward system information
    print(f"Reward system: {'Verdict with GPT-4o' if args.use_verdict else 'Traditional'}")
    if args.use_verdict:
        print("\n=== Using Verdict with GPT-4o for reward scoring ===")
        print("Please set OPENAI_API_KEY environment variable before running")
        print("=================================================\n")
        
    # Run evaluation
    if args.test_all:
        print(f"Running evaluation on all {len(TEST_DISEASE_BANK)} test cases...")
        evaluate_model(model, tokenizer, TEST_DISEASE_BANK, use_verdict=args.use_verdict)
    else:
        print(f"Running evaluation on {args.num_cases} random test cases...")
        evaluate_model(model, tokenizer, num_cases=args.num_cases, use_verdict=args.use_verdict)
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()