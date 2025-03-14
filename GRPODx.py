#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRPODx: Wrapper script to run medical_grpo.py
"""

import os
import sys

if __name__ == "__main__":
    # Simply import and run the main function from medical_grpo
    from medical_grpo import main
    import asyncio
    
    # Extract any arguments or environment variables
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Train a medical diagnosis model with GRPO")
    parser.add_argument("--openai-api-key", help="OpenAI API key for patient simulation (defaults to OPENAI_API_KEY env variable)")
    parser.add_argument("--load-checkpoint", help="Path to load checkpoint from")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--save-steps", type=int, help="Steps between checkpoints")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided either via --openai-api-key argument or OPENAI_API_KEY environment variable")
    
    # Run the async main function
    asyncio.run(main(api_key))