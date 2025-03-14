# GRPODx: Medical Diagnosis Self-Play with Phi-4

GRPODx is a medical diagnosis training system that uses self-play and Generative Reward Policy Optimization (GRPO) to train a diagnosis model using the Phi-4 architecture.

## Overview

The system consists of:

- A **Doctor model** (Phi-4 with LoRA fine-tuning) that learns to diagnose through conversation
- A **Patient simulator** (GPT-4o-mini) that roleplays having a hidden disease
- A **Reward model** (also GPT-4o-mini) that evaluates the doctor's performance

The Doctor model improves through GRPO, where multiple attempts at the same diagnosis scenario are compared to compute advantages.

## Key Features

- **Hidden reasoning**: Doctor responses include `<reason>...</reason>` blocks not shown to the patient
- **Self-play**: Patient model simulates realistic diseases and symptoms
- **Partial credit**: Reward model provides partial scoring for partially correct diagnoses
- **GRPO optimization**: Multiple completions per scenario to compute advantages

## Requirements

- Python 3.8+
- Unsloth
- PyTorch
- OpenAI API key (for patient simulation and rewards)

## Installation

```bash
pip install unsloth torch openai
```

## Usage

```bash
# Using environment variable for API key
export OPENAI_API_KEY=your_api_key
python medical_grpo.py

# Or providing API key directly
python medical_grpo.py --openai-api-key YOUR_API_KEY
```

### Command-line Arguments

- `--openai-api-key`: OpenAI API key for patient simulation (defaults to OPENAI_API_KEY environment variable)
- `--load-checkpoint`: Path to load checkpoint from
- `--max-steps`: Maximum training steps (default: 1000)
- `--save-steps`: Steps between checkpoints (default: 100)
- `--batch-size`: Batch size (default: 2)
- `--lora-rank`: LoRA rank (default: 16)

## How It Works

1. **Conversation Simulation**:
   - Patient picks a hidden disease and presents symptoms
   - Doctor asks questions and provides reasoning
   - Conversation continues for max 5 turns or until diagnosis

2. **Reward Calculation**:
   - After diagnosis, patient reveals the hidden disease
   - Reward model evaluates the doctor's performance (0-1 score)
   - Partial credit given for partially correct diagnoses

3. **GRPO Training**:
   - Multiple completions generated for each scenario
   - Advantages calculated relative to average reward
   - LoRA weights updated using advantage-weighted policy gradient

## License

MIT