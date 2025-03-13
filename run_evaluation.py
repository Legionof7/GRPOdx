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
    max_seq_length = 2048
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
    )
    
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