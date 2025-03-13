# GRPODx: Medical Diagnosis Agent

A medical diagnosis agent that conducts multi-turn conversations with patients and makes diagnoses using Group Relative Policy Optimization (GRPO) reinforcement learning.

## Overview

GRPODx is an autonomous diagnostic agent that:
1. Interacts with patients in a multi-turn dialogue format
2. Dynamically asks relevant questions to gather information
3. Proposes a final diagnosis based on the conversation

This implementation combines:
- Concepts from the AMIE paper (self-play loops and iterative improvement)
- Unsloth's GRPO framework for low-VRAM reinforcement learning
- Dynamic disease generation for unlimited training scenarios

## Key Components

### Architecture

- **Base Model**: Llama 3.1 8B Instruct with QLoRA for fine-tuning
- **Context Window**: 2048 tokens
- **Training Method**: GRPO (Group Relative Policy Optimization)
- **VRAM Usage**: Optimized for ~8GB VRAM

### Self-Play Environment

The implementation uses two agents:
1. **Doctor Agent**: The policy being trained, asks questions and makes diagnoses
2. **Patient Agent**: Simulated agent with "ground truth" disease information

### Dynamic Disease Generation

Instead of using static disease definitions, the system dynamically generates diseases with:
- Medically plausible disease names (e.g., "Chronic Respiratory Syndrome", "Acute Hepatic Disorder")
- Realistic symptom patterns grouped by body systems
- Related disease variants for testing differential diagnosis
- Unlimited training variety to prevent memorization

## Getting Started

### Requirements

```
pip install unsloth vllm
```

#### Optional Dependencies

For using Verdict-based reward scoring:
```
pip install verdict
```

Set your OpenAI API key as an environment variable:
```
export OPENAI_API_KEY='your-api-key'
```

### Training the Model

The main training script is in `GRPODx_implementation.py`. You can train the model using either the command-line interface or directly from Python.

#### Command Line Training

```bash
# Basic training with default parameters
python main.py train

# OR use the training script directly
python run_training.py
```

#### Available Training Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--steps` | int | 500 | Number of training steps |
| `--batch_size` | int | 4 | Batch size for training (number of diseases per batch) |
| `--completions` | int | 6 | Number of completions per scenario for GRPO |
| `--evaluate` | flag | False | Run evaluation after training |
| `--eval_cases` | int | 3 | Number of cases to evaluate if `--evaluate` is set |
| `--verbose` | flag | True | Print detailed training progress including conversations |
| `--quiet` | flag | False | Minimize output during training (overrides `--verbose`) |
| `--llm_patient` | flag | False | Use LLM-based patient simulator instead of rule-based |
| `--no_verdict` | flag | False | Disable Verdict-based reward scoring (uses traditional rewards) |

#### Example Training Commands

```bash
# Training with custom parameters
python main.py train --steps 1000 --batch_size 8 --completions 8

# Training with LLM-based patient simulator
python main.py train --llm_patient

# Short training with evaluation
python main.py train --steps 100 --evaluate --eval_cases 5

# Training with Verdict-based reward scoring (requires OpenAI API key)
python main.py train --steps 200 # Verdict is enabled by default, use --no_verdict to disable

# Quiet mode with minimal output
python main.py train --quiet
```

#### Using Python API

```python
from GRPODx_implementation import train_grpodx

# Train the model with default parameters
model, tokenizer = train_grpodx()

# Train with custom parameters
model, tokenizer = train_grpodx(
    num_steps=1000,
    batch_size=8,
    completions_per_scenario=8,
    verbose=True,
    use_llm_patient=True
)
```

### Interactive Mode

After training, you can interact with the model in different ways:

```bash
# Start an interactive diagnostic session where you provide your symptoms
python main.py interactive

# Run in simulation mode with a randomly generated virtual patient
python main.py interactive --simulate

# Use a specific model
python main.py interactive --model "meta-llama/Llama-2-7b-chat-hf"
```

Or directly from Python:

```python
from GRPODx_implementation import interactive_diagnosis

# Start an interactive diagnostic session
interactive_diagnosis(model, tokenizer)

# Start with a simulated patient
interactive_diagnosis(model, tokenizer, simulation_mode=True)
```

### Evaluation

To evaluate the model's performance:

```bash
# Run evaluation with 3 random test cases
python main.py evaluate

# Run evaluation with all predefined test cases
python main.py evaluate --test_all

# Run evaluation with a specific number of test cases
python main.py evaluate --num_cases 10
```

Or from Python:

```python
from test_scenarios import evaluate_model

# Run evaluation on random test cases
results = evaluate_model(model, tokenizer)

# Run evaluation on specific test cases
from disease_generator import generate_disease_batch
test_diseases = generate_disease_batch(5)
results = evaluate_model(model, tokenizer, test_cases=test_diseases)
```

## Key Features

1. **Multi-turn Dialogue**: Conducts complete diagnostic conversations
2. **GRPO Reinforcement Learning**: Uses group advantage to improve policy
3. **Low VRAM Training**: Optimized for consumer-grade GPUs
4. **Structured Output**: Uses a reasoning-question format for better diagnostics
5. **Dynamic Disease Generation**: Creates unlimited training scenarios
6. **Simulation Mode**: Test the model with virtual patients
7. **Enhanced Evaluation**: Measures accuracy, efficiency, and symptom coverage
8. **Verdict-based Rewards**: Optional advanced reward scoring using GPT-4o for better diagnostic quality assessment

## Implementation Notes

- The system uses 4-bit quantization to reduce VRAM usage
- GRPO trains with multiple completions per scenario (6 by default)
- Training rewards consider diagnosis accuracy, question repetition, and symptom coverage
- The disease generator creates medically plausible conditions with realistic symptom patterns
- Disease caching creates a small memory of previously seen conditions for more stable training
- Verdict-based rewards use GPT-4o to score diagnostic quality on a 0-1 scale with detailed rubrics
- The system gracefully falls back to traditional rewards if Verdict or OpenAI API is unavailable

## Extension Possibilities

1. Add a critic agent for feedback on the doctor's conversation
2. Implement a differential diagnosis mode that suggests multiple conditions
3. Add severity scoring based on symptom combinations
4. Create a specialized model for specific medical domains
5. Implement a web interface with speech input/output
6. Add a visualization component to show diagnostic reasoning paths

## References

- [Unsloth GRPO Documentation](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl)
- AMIE: An Adaptive Model for Information Extraction
- Llama 3.1 Model Documentation