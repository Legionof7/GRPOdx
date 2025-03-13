# GRPODx: Medical Diagnosis Agent (Rewrite)

A medical diagnosis agent that conducts multi-turn conversations with patients and makes diagnoses using Group Relative Policy Optimization (GRPO) reinforcement learning.

## Project Status

This repository is being rewritten from scratch to create a more robust implementation. The previous implementation has been preserved in the `backup_implementation` branch.

## Overview

GRPODx is an autonomous diagnostic agent that:
1. Interacts with patients in a multi-turn dialogue format
2. Dynamically asks relevant questions to gather information
3. Proposes a final diagnosis based on the conversation

The project uses:
- Concepts from the AMIE paper (self-play loops and iterative improvement)
- Group Relative Policy Optimization (GRPO) framework for reinforcement learning
- Dynamic disease generation for unlimited training scenarios

## Project Plan

The rewrite will focus on:

1. Cleaner architecture with better separation of concerns
2. More robust handling of different model types and configurations
3. Improved error handling and fallback mechanisms
4. Better documentation and code organization
5. Comprehensive testing framework

## Resources

Documentation for the technical approach and reference examples are available in the `docs` folder:

- `techspec.txt`: Technical specification for the project
- `unsloth.txt`: Documentation for Unsloth integration
- `verdict_example.py`: Example of Verdict integration for reward calculation
- `verdictdocs.txt`: Documentation for the Verdict integration

## References

- [Unsloth GRPO Documentation](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl)
- AMIE: An Adaptive Model for Information Extraction
- Llama 3.1 Model Documentation