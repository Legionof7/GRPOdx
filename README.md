# GRPODx

**Autonomous Diagnostic Agent using Group Relative Policy Optimization**

GRPODx is an autonomous diagnostic chatbot that uses reinforcement learning (specifically Group Relative Policy Optimization) to perform medical diagnosis through natural conversation.

## Overview

This project implements a diagnostic system that:

1. Conducts multi-turn dialogue with a synthetic patient
2. Asks relevant questions to gather information
3. Provides a final diagnosis
4. Uses GRPO to improve diagnostic accuracy through reinforcement learning

## Technical Implementation

GRPODx combines:

- **Gemma 3 1B** as the base language model
- **Unsloth's GRPO implementation** for reinforcement learning
- **QLoRA fine-tuning** for efficient training with limited GPU resources
- **vLLM** for fast inference
- **GPT-4o-mini** for patient simulation and reward calculation

## Architecture

The system consists of several key components:

1. **Doctor Agent**: The LLM-based policy being trained via GRPO
2. **Patient Agent**: Uses GPT-4o-mini to simulate realistic patient responses based on disease symptoms
3. **Disease Bank**: A collection of diseases with their associated symptoms
4. **GPT-4o-mini Reward System**:
   - Uses GPT-4o-mini to evaluate diagnostic correctness
   - Provides numerical scores from 0.0 to 1.0
   - Rewards both accuracy and structured reasoning
5. **GRPO Trainer**: Coordinates the training process using multiple completions per scenario

## Features

- Multi-turn dialogue between doctor and patient
- Structured reasoning with XML-based format
- Final diagnosis extraction and evaluation
- Partial credit for close diagnostic matches
- 4-bit quantization for low VRAM usage (~7-8GB)
- Interactive chat mode for real user inputs

## Usage

### Installation

```bash
pip install unsloth vllm torch openai
```

### Training

The script supports multiple modes of operation:

```bash
# Train the model (default)
python GRPODx.py --train

# Test the model with simulated patients
python GRPODx.py --test --num-tests 5 --use-gpt-patient

# Run interactive chat mode to talk with the AI doctor
python GRPODx.py --interact

# Run multiple operations
python GRPODx.py --train --test --interact
```

#### Training Mode
When training, the script will:
1. Load the disease definitions from `diseases.json`
2. Initialize the model with LoRA adapters
3. Create a training dataset with random disease scenarios
4. Train using GRPO for the specified number of steps
5. Save the trained model to the specified output path

#### Testing Mode
When testing, the script will:
1. Run diagnostic episodes with simulated patients
2. Evaluate the model's diagnostic accuracy
3. Optionally use GPT-4o-mini for more realistic patient responses

#### Interactive Mode
In interactive mode, you can:
1. Chat directly with the AI doctor as a patient
2. Describe your symptoms and answer the doctor's questions
3. Receive a final diagnosis with structured reasoning

### Inference

The trained model can be used for interactive diagnosis:

```python
from GRPODx import test_model, DiseaseBank
import torch
import os
from unsloth import FastLanguageModel

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True
)

# Load disease bank
disease_bank = DiseaseBank("diseases.json")

# Test with interactive diagnosis (use_gpt_patient=False to use rule-based patient if no API key)
test_model(model, tokenizer, disease_bank, num_tests=3, use_gpt_patient=True)
```

## Customization

- **Disease Definitions**: Edit `diseases.json` to add or modify diseases and their symptoms
- **Model Parameters**: Adjust LoRA rank, learning rate, etc. in the script
- **Training Settings**: Modify the number of steps, batch size, etc.
- **Reward Functions**: Customize the reward functions for different training objectives

## Limitations

- Limited to pre-defined diseases and symptoms 
- Performance depends on the quality of the base model
- Simplified patient responses compared to real-world scenarios

## Future Work

- Add more complex diseases with overlapping symptoms
- Implement more sophisticated reward functions
- Add medical history and demographic information
- Support multi-modal inputs (e.g., medical images)
- Develop a more realistic patient simulator

## License

This project is for educational purposes. The medical diagnoses provided by this model should not be used for actual medical decisions.

## Acknowledgments

This project builds on the Unsloth library and the concept of Group Relative Policy Optimization for reinforcement learning with LLMs.