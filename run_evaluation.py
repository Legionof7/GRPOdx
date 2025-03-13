"""
Evaluation script for GRPODx Medical Diagnosis Agent.

This script loads a trained model and runs evaluation on test scenarios.
"""

from unsloth import FastLanguageModel
from test_scenarios import evaluate_model, TEST_DISEASE_BANK
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPODx model")
    parser.add_argument("--num_cases", type=int, default=3, 
                      help="Number of test cases to evaluate")
    parser.add_argument("--test_all", action="store_true",
                      help="Test on all available disease scenarios")
    args = parser.parse_args()
    
    print("Loading GRPODx Medical Diagnosis Agent...")
    
    # Load model parameters (matching training parameters)
    max_seq_length = 4096  # Updated to match the new context window
    lora_rank = 8
    
    # Load base model - using unsloth GGUF 70B model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.3-70B-Instruct-GGUF",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.85,  # Optimized setting
    )
    
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
    
    # Run evaluation
    if args.test_all:
        print(f"Running evaluation on all {len(TEST_DISEASE_BANK)} test cases...")
        evaluate_model(model, tokenizer, TEST_DISEASE_BANK)
    else:
        print(f"Running evaluation on {args.num_cases} random test cases...")
        evaluate_model(model, tokenizer, num_cases=args.num_cases)
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()