"""
Configuration file for GRPODx Medical Diagnosis Agent.

This file contains configuration parameters that can be modified to customize the model.
"""

# Model Configuration
MODEL_CONFIG = {
    # Base model to use - will attempt these models in order until one works
    "model_options": [
        "unsloth/gemma-3-1b-it-bnb-4bit",          # Primary model - Gemma 3 1B (BNB 4-bit instruct-tuned)
        "unsloth/gemma-3-27b-it-GGUF",             # Alternative 27B model (GGUF format)
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit", # Alternative 70B model (BNB 4-bit format)
        "meta-llama/meta-Llama-3.1-8B-Instruct",   # Fallback - Llama 3.1 8B
        "unsloth/Llama-3.1-8B-Instruct",           # Unsloth mirror
        "unsloth/llama-3-8b-instruct",             # Alternative naming
        "Qwen/Qwen2-7B-Instruct",                  # Qwen alternative
        "mistralai/Mistral-7B-Instruct-v0.2",      # Mistral alternative
        "meta-llama/Llama-2-7b-chat-hf",           # Llama 2 fallback
        "unsloth/Qwen2.5-1.5B-Instruct"            # Smallest model option
    ],
    
    # Legacy model name for backwards compatibility
    "model_name": "unsloth/gemma-3-1b-it-bnb-4bit",
    
    # Model parameters
    "max_seq_length": 8192,  # Doubled from 4096 to allow much longer conversations
    "load_in_4bit": True,  # Using 4-bit quantization for smaller model
    "fast_inference": True,
    "gpu_memory_utilization": 0.80,  # Adjusted for 1B model
    "rope_scaling": {"type": "dynamic", "factor": 4.0},  # Doubled factor for expanded context window
    
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
    "learning_rate": 1e-6,        # Lower for 70B model
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,  # Increased for 70B model
    "max_prompt_length": 1024,
    "max_grad_norm": 0.1,
    "output_dir": "outputs",
}

# Diagnostic Configuration
DIAGNOSTIC_CONFIG = {
    "max_turns": 24,  # Doubled from 12 to allow much more conversation turns
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 1024,  # Increased from 768 to allow longer responses
}

# System Prompt (can be customized)
SYSTEM_PROMPT = """
You are a medical diagnostic assistant. Your goal is to diagnose the patient's condition by asking relevant questions.
Follow these rules:
1. Ask one question at a time about symptoms.
2. Don't repeat questions you've already asked.
3. When you have enough information, provide your final diagnosis in the format "Final diagnosis: [SPECIFIC DISEASE NAME]".
4. Be concise and professional.
5. You must provide a diagnosis by the 20th question at the latest, even if uncertain.
6. If it's the 16th question or later, start considering your final diagnosis.

Important diagnosis guidelines:
- Always give a specific disease name, not a general description of symptoms
- Never say "Final diagnosis: of a condition..." or other vague statements
- If uncertain, still provide your best specific diagnosis based on the symptoms
- Good examples: "Final diagnosis: Chronic Bronchitis" or "Final diagnosis: Migraine Headache"
- Bad examples: "Final diagnosis: some kind of respiratory issue" or "Final diagnosis: a neurological condition"
- Submit your diagnosis as soon as you're confident - earlier correct diagnoses receive higher rewards
- The exact format "Final diagnosis: [SPECIFIC DISEASE NAME]" is required for the verification system

Format your response as:
<reasoning>
Your internal reasoning about the patient's condition based on symptoms revealed so far.
If this is the 16th question or later, you should be formulating your final diagnosis.
Consider which specific disease best matches the patient's symptoms.
</reasoning>
<question>
Your next question to the patient.
</question>

OR, when you're ready to provide a diagnosis:

<reasoning>
Your internal reasoning explaining why you believe this is the correct diagnosis.
Summarize the key symptoms and how they point to this specific disease.
</reasoning>
<diagnosis>
Final diagnosis: [SPECIFIC DISEASE NAME]
</diagnosis>
"""