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
    parser.add_argument("--llm_patient", action="store_true",
                      help="Use LLM-based patient simulator instead of rule-based")
    parser.add_argument("--use_verdict", action="store_true", default=True,
                      help="Use Verdict-based reward scoring with GPT-4o")
    parser.add_argument("--no_verdict", action="store_true",
                      help="Disable Verdict-based reward scoring")
    args = parser.parse_args()
    
    # --no_verdict overrides --use_verdict
    if args.no_verdict:
        args.use_verdict = False
    
    # Set verbosity (--quiet overrides --verbose)
    verbose = args.verbose and not args.quiet
    
    print(f"Starting GRPODx training for {args.steps} steps...")
    print(f"Using batch size {args.batch_size} with {args.completions} completions per scenario")
    print(f"Verbose output: {'Enabled' if verbose else 'Disabled'}")
    print(f"Patient type: {'LLM-based' if args.llm_patient else 'Rule-based'}")
    print(f"Reward system: {'Verdict with GPT-4o' if args.use_verdict else 'Traditional'}")
    
    # Display Verdict information if enabled
    if args.use_verdict:
        print("\n=== Using Verdict with GPT-4o for reward scoring ===")
        print("Please set OPENAI_API_KEY environment variable before running")
        print("=================================================\n")
    
    # Train the model
    model, tokenizer = train_grpodx(
        num_steps=args.steps,
        batch_size=args.batch_size,
        completions_per_scenario=args.completions,
        verbose=verbose,
        use_llm_patient=args.llm_patient,
        use_verdict=args.use_verdict
    )
    
    print("Training complete!")
    
    # Evaluate if requested
    if args.evaluate:
        print(f"\nEvaluating model on {args.eval_cases} test cases...")
        evaluate_model(model, tokenizer, num_cases=args.eval_cases, use_verdict=args.use_verdict)

if __name__ == "__main__":
    main()