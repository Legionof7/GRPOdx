"""
Configuration file for GRPODx Medical Diagnosis Agent.

This file contains configuration parameters that can be modified to customize the model.
"""

# Model Configuration
MODEL_CONFIG = {
    # Base model to use - will attempt these models in order until one works
    "model_options": [
        "meta-llama/meta-Llama-3.1-8B-Instruct",  # First choice
        "unsloth/Llama-3.1-8B-Instruct",          # Unsloth mirror
        "unsloth/llama-3-8b-instruct",            # Alternative naming
        "unsloth/Qwen2.5-7B-Instruct",            # Alternative model
        "meta-llama/Llama-2-7b-chat-hf",          # Llama 2 fallback
        "unsloth/Qwen2.5-1.5B-Instruct"           # Smallest model option
    ],
    
    # Legacy model name for backwards compatibility
    "model_name": "meta-llama/meta-Llama-3.1-8B-Instruct",
    
    # Model parameters
    "max_seq_length": 2048,
    "load_in_4bit": True,  # Set to False for 16-bit training (higher VRAM usage)
    "fast_inference": True,
    
    # LoRA parameters
    "lora_rank": 8,
    "lora_alpha": 8,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "use_gradient_checkpointing": "unsloth",  # Options: "unsloth", True, False
}

# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_prompt_length": 512,
    "max_grad_norm": 0.1,
    "output_dir": "outputs",
}

# Diagnostic Configuration
DIAGNOSTIC_CONFIG = {
    "max_turns": 8,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 512,
}

# System Prompt (can be customized)
SYSTEM_PROMPT = """
You are a medical diagnostic assistant. Your goal is to diagnose the patient's condition by asking relevant questions.
Follow these rules:
1. Ask one question at a time about symptoms.
2. Don't repeat questions you've already asked.
3. When you have enough information, provide your final diagnosis in the format "Final diagnosis: [DISEASE]".
4. Be concise and professional.

Format your response as:
<reasoning>
Your internal reasoning about the patient's condition based on symptoms revealed so far.
</reasoning>
<question>
Your next question to the patient OR your final diagnosis.
</question>
"""