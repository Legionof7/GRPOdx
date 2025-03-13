"""
Main entry point for GRPODx Medical Diagnosis Agent.

This script provides a command-line interface to:
1. Train the model
2. Evaluate the model
3. Run interactive diagnosis
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="GRPODx Medical Diagnosis Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--steps", type=int, default=500, 
                           help="Number of training steps")
    train_parser.add_argument("--batch_size", type=int, default=4,
                           help="Batch size for training")
    train_parser.add_argument("--completions", type=int, default=6,
                           help="Number of completions per scenario for GRPO")
    train_parser.add_argument("--evaluate", action="store_true",
                           help="Run evaluation after training")
    train_parser.add_argument("--eval_cases", type=int, default=3,
                           help="Number of cases to evaluate if --evaluate is set")
    train_parser.add_argument("--verbose", action="store_true", default=True,
                           help="Print detailed training progress including conversations")
    train_parser.add_argument("--quiet", action="store_true",
                           help="Minimize output during training")
    train_parser.add_argument("--llm_patient", action="store_true",
                           help="Use LLM-based patient simulator instead of rule-based")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--num_cases", type=int, default=3, 
                          help="Number of test cases to evaluate")
    eval_parser.add_argument("--test_all", action="store_true",
                          help="Test on all available disease scenarios")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive diagnostic session")
    interactive_parser.add_argument("--simulate", action="store_true",
                                help="Run in simulation mode with a virtual patient")
    interactive_parser.add_argument("--model", type=str,
                                help="Model to use (defaults to configured model)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the appropriate script based on command
    if args.command == "train":
        train_cmd = [sys.executable, "run_training.py"]
        train_cmd.extend(["--steps", str(args.steps)])
        train_cmd.extend(["--batch_size", str(args.batch_size)])
        train_cmd.extend(["--completions", str(args.completions)])
        if args.evaluate:
            train_cmd.append("--evaluate")
            train_cmd.extend(["--eval_cases", str(args.eval_cases)])
        if args.verbose:
            train_cmd.append("--verbose")
        if args.quiet:
            train_cmd.append("--quiet")
        if args.llm_patient:
            train_cmd.append("--llm_patient")
        os.execv(sys.executable, train_cmd)
    
    elif args.command == "evaluate":
        eval_cmd = [sys.executable, "run_evaluation.py"]
        eval_cmd.extend(["--num_cases", str(args.num_cases)])
        if args.test_all:
            eval_cmd.append("--test_all")
        os.execv(sys.executable, eval_cmd)
    
    elif args.command == "interactive":
        interactive_cmd = [sys.executable, "run_interactive.py"]
        if args.simulate:
            interactive_cmd.append("--simulate")
        if args.model:
            interactive_cmd.extend(["--model", args.model])
        os.execv(sys.executable, interactive_cmd)

if __name__ == "__main__":
    main()