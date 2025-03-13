"""
Interactive script for GRPODx Medical Diagnosis Agent.

This script loads a trained model and runs an interactive diagnostic session.
"""

import argparse
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
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        fast_inference=MODEL_CONFIG["fast_inference"],
    )
    
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