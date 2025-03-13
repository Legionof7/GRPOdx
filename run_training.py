"""
Training script for GRPODx Medical Diagnosis Agent.

This script runs the training process and optionally evaluates the model.
"""

import argparse
from GRPODx_implementation import train_grpodx
from test_scenarios import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train GRPODx model")
    parser.add_argument("--steps", type=int, default=500, 
                      help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training")
    parser.add_argument("--completions", type=int, default=6,
                      help="Number of completions per scenario for GRPO")
    parser.add_argument("--evaluate", action="store_true",
                      help="Run evaluation after training")
    parser.add_argument("--eval_cases", type=int, default=3,
                      help="Number of cases to evaluate if --evaluate is set")
    parser.add_argument("--verbose", action="store_true", default=True,
                      help="Print detailed training progress")
    parser.add_argument("--quiet", action="store_true",
                      help="Minimize output during training")
    args = parser.parse_args()
    
    # Set verbosity (--quiet overrides --verbose)
    verbose = args.verbose and not args.quiet
    
    print(f"Starting GRPODx training for {args.steps} steps...")
    print(f"Using batch size {args.batch_size} with {args.completions} completions per scenario")
    print(f"Verbose output: {'Enabled' if verbose else 'Disabled'}")
    
    # Train the model
    model, tokenizer = train_grpodx(
        num_steps=args.steps,
        batch_size=args.batch_size,
        completions_per_scenario=args.completions,
        verbose=verbose
    )
    
    print("Training complete!")
    
    # Evaluate if requested
    if args.evaluate:
        print(f"\nEvaluating model on {args.eval_cases} test cases...")
        evaluate_model(model, tokenizer, num_cases=args.eval_cases)

if __name__ == "__main__":
    main()