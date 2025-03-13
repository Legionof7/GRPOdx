"""
Configuration file for GRPODx Medical Diagnosis Agent.

This file contains configuration parameters that can be modified to customize the model.
"""

# Model Configuration
MODEL_CONFIG = {
    # Base model to use - will attempt these models in order until one works
    "model_options": [
        "unsloth/Llama-3.3-70B-Instruct-GGUF",    # Primary large model
        "meta-llama/meta-Llama-3.1-8B-Instruct",  # Secondary choice
        "unsloth/Llama-3.1-8B-Instruct",          # Unsloth mirror
        "unsloth/llama-3-8b-instruct",            # Alternative naming
        "unsloth/Qwen2.5-7B-Instruct",            # Alternative model
        "meta-llama/Llama-2-7b-chat-hf",          # Llama 2 fallback
        "unsloth/Qwen2.5-1.5B-Instruct"           # Smallest model option
    ],
    
    # Legacy model name for backwards compatibility
    "model_name": "unsloth/Llama-3.3-70B-Instruct-GGUF",
    
    # Model parameters
    "max_seq_length": 4096,  # Increased from 2048 to allow longer conversations
    "load_in_4bit": True,  # Set to False for 16-bit training (higher VRAM usage)
    "fast_inference": True,
    "gpu_memory_utilization": 0.9,  # Increased for 70B model
    
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
    "learning_rate": 2e-6,        # Reduced for 70B model
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,  # Increased for 70B model
    "max_prompt_length": 1024,
    "max_grad_norm": 0.1,
    "output_dir": "outputs",
}

# Diagnostic Configuration
DIAGNOSTIC_CONFIG = {
    "max_turns": 12,  # Increased from 8 to allow more conversation turns
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 768,  # Increased from 512 to allow longer responses
}

# System Prompt (can be customized)
SYSTEM_PROMPT = """
You are a medical diagnostic assistant. Your goal is to diagnose the patient's condition by asking relevant questions.
Follow these rules:
1. Ask one question at a time about symptoms.
2. Don't repeat questions you've already asked.
3. When you have enough information, provide your final diagnosis in the format "Final diagnosis: [SPECIFIC DISEASE NAME]".
4. Be concise and professional.
5. You must provide a diagnosis by the 10th question at the latest, even if uncertain.
6. If it's the 8th question or later, start considering your final diagnosis.

Important diagnosis guidelines:
- Always give a specific disease name, not a general description of symptoms
- Never say "Final diagnosis: of a condition..." or other vague statements
- If uncertain, still provide your best specific diagnosis based on the symptoms
- Good examples: "Final diagnosis: Chronic Bronchitis" or "Final diagnosis: Migraine Headache"
- Bad examples: "Final diagnosis: some kind of respiratory issue" or "Final diagnosis: a neurological condition"

Format your response as:
<reasoning>
Your internal reasoning about the patient's condition based on symptoms revealed so far.
If this is the 8th question or later, you should be formulating your final diagnosis.
Consider which specific disease best matches the patient's symptoms.
</reasoning>
<question>
Your next question to the patient OR your final diagnosis.
If this is the 10th question or later, you MUST provide a final diagnosis with a specific disease name.
</question>
"""