# -*- coding: utf-8 -*-
"""
GRPODx: Medical Diagnosis Agent using GRPO (Group Relative Policy Optimization)

This implementation follows the technical specification to create a medical diagnostic 
agent that can conduct multi-turn conversations with patients and make diagnoses.
"""

# Install dependencies
# !pip install unsloth vllm

# Import necessary libraries
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import random
import json
import re
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# Import disease generator for dynamic disease creation
from disease_generator import generate_random_disease, generate_disease_batch, generate_related_diseases

# Generate example diseases for reference
DISEASE_EXAMPLES = [
    {
        "disease_name": "Influenza",
        "symptoms": {
            "fever": True,
            "cough": True,
            "headache": True,
            "sore_throat": True,
            "body_aches": True,
            "fatigue": True,
            "runny_nose": True,
            "chills": False,
            "nausea": False,
            "vomiting": False,
            "diarrhea": False,
            "shortness_of_breath": False,
            "chest_pain": False,
            "rash": False
        }
    },
    {
        "disease_name": "Common Cold",
        "symptoms": {
            "fever": False,
            "cough": True,
            "headache": False,
            "sore_throat": True,
            "body_aches": False,
            "fatigue": True,
            "runny_nose": True,
            "chills": False,
            "nausea": False,
            "vomiting": False,
            "diarrhea": False,
            "shortness_of_breath": False,
            "chest_pain": False,
            "rash": False
        }
    }
]

# Function to get a disease (either generate a new one or retrieve from cache)
def get_disease(cache=None, use_cache_probability=0.2):
    """
    Get a disease for training or evaluation
    
    Args:
        cache: Optional list to use as a cache of previously generated diseases
        use_cache_probability: Probability of using a cached disease if available
    
    Returns:
        A disease dictionary with name and symptoms
    """
    # If cache exists and not empty, potentially use a cached disease
    if cache and len(cache) > 0 and random.random() < use_cache_probability:
        return random.choice(cache)
    
    # Otherwise generate a new disease
    return generate_random_disease()

# Define prompt formats
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

def format_conversation(conversation_history):
    """Format conversation history for model input"""
    formatted_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    for entry in conversation_history:
        formatted_messages.append(entry)
    
    return formatted_messages

# Patient agent simulation
def patient_response(question, disease_info):
    """Simulate a patient response based on the disease info"""
    symptoms = disease_info["symptoms"]
    
    # Check if the question contains known symptoms
    response = "I'm not sure what you mean."
    
    for symptom, has_symptom in symptoms.items():
        # Simple keyword matching for symptoms in the question
        if symptom.replace("_", " ") in question.lower():
            if has_symptom:
                response = f"Yes, I have {symptom.replace('_', ' ')}."
            else:
                response = f"No, I don't have {symptom.replace('_', ' ')}."
            break
    
    return response

# Extract parts from model output
def extract_reasoning(text):
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    match = re.search(reasoning_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_question(text):
    question_pattern = r"<question>(.*?)</question>"
    match = re.search(question_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_diagnosis(text):
    diagnosis_pattern = r"Final diagnosis: ([A-Za-z\s\-]+)"
    match = re.search(diagnosis_pattern, text)
    if match:
        return match.group(1).strip()
    return None

# Episode simulation
def run_episode(model, tokenizer, disease_info=None, max_turns=5):
    """
    Run a complete diagnostic episode
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        disease_info: Optional disease info. If None, a new disease will be generated
        max_turns: Maximum number of conversation turns
        
    Returns:
        conversation, final_diagnosis, reward
    """
    conversation = []
    
    # Generate a disease if none provided
    if disease_info is None:
        disease_info = generate_random_disease()
    
    # Initial patient message with a primary symptom
    primary_symptoms = [s for s, has in disease_info["symptoms"].items() if has]
    if primary_symptoms:
        initial_symptom = random.choice(primary_symptoms)
        initial_message = f"I'm not feeling well. I have {initial_symptom.replace('_', ' ')}."
    else:
        initial_message = "I'm not feeling well."
    
    conversation.append({"role": "user", "content": initial_message})
    
    final_diagnosis = None
    questions_asked = []
    
    # Diagnostic conversation loop
    for _ in range(max_turns):
        # Format conversation for model input
        formatted_conv = format_conversation(conversation)
        prompt = tokenizer.apply_chat_template(formatted_conv, tokenize=False, add_generation_prompt=True)
        
        # Generate doctor's response
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
        )
        
        response = model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
        )[0].outputs[0].text
        
        # Extract parts
        question_text = extract_question(response)
        
        if not question_text:
            # If no valid question format, skip this turn
            continue
        
        # Check if this is a final diagnosis
        diagnosis = extract_diagnosis(question_text)
        
        # Add doctor's response to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # Track questions for repetition detection
        questions_asked.append(question_text)
        
        if diagnosis:
            final_diagnosis = diagnosis
            break
        
        # Generate patient's response
        patient_reply = patient_response(question_text, disease_info)
        conversation.append({"role": "user", "content": patient_reply})
    
    # Calculate reward
    reward = 0.0
    
    # Exact match reward
    if final_diagnosis:
        if final_diagnosis.lower() == disease_info["disease_name"].lower():
            reward = 1.0
        else:
            # Partial match reward - if disease name contains parts of real diagnosis
            disease_words = set(disease_info["disease_name"].lower().split())
            diagnosis_words = set(final_diagnosis.lower().split())
            common_words = disease_words.intersection(diagnosis_words)
            
            if len(common_words) > 0 and len(common_words) >= len(disease_words) / 3:
                reward = 0.3  # Partial credit for related diagnosis
    
    # Penalize for question repetition
    unique_questions = set([q.lower() for q in questions_asked])
    if len(questions_asked) > len(unique_questions):
        repetition_penalty = 0.1 * (len(questions_asked) - len(unique_questions))
        reward = max(0, reward - repetition_penalty)
    
    return conversation, final_diagnosis, reward, disease_info

# Prepare dataset for GRPO
def generate_training_batch(model, tokenizer, batch_size=4, completions_per_scenario=6):
    """
    Generate a batch of training data for GRPO using dynamically generated diseases
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        batch_size: Number of different disease scenarios to generate
        completions_per_scenario: Number of completions per disease scenario for GRPO
    
    Returns:
        Dataset object containing training data
    """
    training_data = []
    disease_cache = []  # Cache of generated diseases for potential reuse
    
    for _ in range(batch_size):
        # Generate a new disease scenario with 80% probability, or use cached one with 20% probability
        disease_scenario = get_disease(cache=disease_cache, use_cache_probability=0.2)
        
        # Add to cache for potential reuse
        disease_cache.append(disease_scenario)
        
        # Keep cache size reasonable
        if len(disease_cache) > 20:
            disease_cache = disease_cache[-20:]
        
        # Occasionally generate related diseases (10% chance)
        if random.random() < 0.1 and len(disease_cache) > 0:
            base_disease = random.choice(disease_cache)
            related_disease = generate_related_diseases(base_disease, 1)[0]
            disease_scenario = related_disease
            disease_cache.append(related_disease)
        
        episode_data = []
        for _ in range(completions_per_scenario):
            # Run an episode with the same disease
            conversation, diagnosis, reward, _ = run_episode(model, tokenizer, disease_scenario)
            
            # Format for GRPO training
            formatted_conv = format_conversation(conversation)
            prompt = tokenizer.apply_chat_template(formatted_conv[:-1], tokenize=False, add_generation_prompt=True)
            
            # Add to episode data
            episode_data.append({
                "prompt": prompt,
                "completion": conversation[-1]["content"],
                "reward": reward,
                "disease_name": disease_scenario["disease_name"],
                "disease_scenario": disease_scenario  # Pass full disease info
            })
        
        # Calculate group relative advantage
        avg_reward = sum(item["reward"] for item in episode_data) / len(episode_data)
        
        for item in episode_data:
            item["advantage"] = item["reward"] - avg_reward
            # Include reward function metadata to be used by the diagnosis_reward_func
            item["reward_meta"] = {
                "disease_name": item["disease_name"],
                "disease_cache": disease_cache
            }
            training_data.append(item)
    
    return Dataset.from_list(training_data)

# Main training function
def train_grpodx(num_steps=500, batch_size=4, completions_per_scenario=6):
    """Main training function for GRPODx"""
    # Set up model parameters
    max_seq_length = 2048
    lora_rank = 8
    
    # Create a global disease cache that can be shared with reward functions
    disease_cache = []
    
    # Load model with fallbacks for different environments
    model_options = [
        "meta-llama/meta-Llama-3.1-8B-Instruct",  # First choice
        "unsloth/Llama-3.1-8B-Instruct",          # Unsloth mirror
        "unsloth/llama-3-8b-instruct",            # Alternative naming
        "unsloth/Qwen2.5-7B-Instruct",            # Alternative model
        "meta-llama/Llama-2-7b-chat-hf",          # Llama 2 fallback
        "unsloth/Qwen2.5-1.5B-Instruct"           # Smallest model option
    ]
    
    # Try models in order until one works
    for model_name in model_options:
        try:
            print(f"Attempting to load model: {model_name}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                fast_inference=True,
                max_lora_rank=lora_rank,
                gpu_memory_utilization=0.6,
            )
            print(f"Successfully loaded: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            continue
    else:
        raise ValueError("Could not load any of the available models. Please check your environment.")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Prepare GRPO configuration
    max_prompt_length = 512
    
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=completions_per_scenario,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        max_steps=num_steps,
        save_steps=num_steps // 5,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs",
    )
    
    # Generate initial training dataset
    initial_dataset = generate_training_batch(model, tokenizer, batch_size, completions_per_scenario)
    
    # Create custom reward function
    def diagnosis_reward_func(prompts, completions, **kwargs):
        """Custom reward function for GRPODx"""
        rewards = []
        
        # Extract disease names from dataset 
        disease_names = []
        try:
            # First try to get disease names from the training dataset
            if hasattr(initial_dataset, 'features') and 'disease_name' in initial_dataset.features:
                disease_names = list(set(initial_dataset['disease_name']))
        except Exception as e:
            print(f"Warning: Could not extract disease names from dataset: {e}")
        
        # Add example disease names as fallback
        example_diseases = [d["disease_name"] for d in DISEASE_EXAMPLES]
        disease_names.extend(example_diseases)
        
        # We can't reference the trainer here as it doesn't exist yet
        # Just use the disease_cache directly
            
        # Add any generated diseases
        if len(disease_cache) > 0:
            disease_names.extend([d.get("disease_name", "") for d in disease_cache])
        
        # Ensure all disease names are strings
        disease_names = [str(name) for name in disease_names if name]
        
        for completion in completions:
            # Handle different completion formats - could be a string or a dict
            if isinstance(completion, list) and len(completion) > 0:
                if isinstance(completion[0], dict) and "content" in completion[0]:
                    response = completion[0]["content"]
                else:
                    response = str(completion[0])
            elif isinstance(completion, dict) and "content" in completion:
                response = completion["content"]
            else:
                response = str(completion)
            
            # Get diagnosis from response
            diagnosis = extract_diagnosis(extract_question(response))
            
            # Basic reward - can be expanded with partial matching etc.
            reward = 0.0
            if diagnosis:
                # Base reward for providing any diagnosis in correct format
                reward = 0.5
                
                # Check against known disease names
                for disease_name in disease_names:
                    if disease_name.lower() in diagnosis.lower():
                        reward = 1.0
                        break
                
                # If we see XML tags, that's good formatting
                if "<reasoning>" in response and "</reasoning>" in response:
                    reward += 0.2
                
                if "<question>" in response and "</question>" in response:
                    reward += 0.2
            
            rewards.append(reward)
        
        return rewards
    
    # Setup GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[diagnosis_reward_func],
        args=training_args,
        train_dataset=initial_dataset,
    )
    
    # Attach disease cache to trainer for reference (won't be used directly by GRPOTrainer)
    trainer.disease_cache = disease_cache
    
    # Train the model
    trainer.train()
    
    # Save the trained model
    model.save_lora("grpodx_model")
    
    return model, tokenizer

# Interactive diagnosis mode
def interactive_diagnosis(model, tokenizer, simulation_mode=False):
    """
    Run interactive diagnostic session with the trained model
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        simulation_mode: If True, will simulate a patient with a random disease
        
    Returns:
        None
    """
    conversation = []
    
    print("GRPODx Medical Diagnostic Assistant")
    print("===================================")
    
    # For simulation mode, generate a disease and simulate patient responses
    simulated_disease = None
    if simulation_mode:
        simulated_disease = generate_random_disease()
        print(f"[Simulation Mode - Patient has: {simulated_disease['disease_name']}]")
        print(f"[Symptoms: {', '.join([s.replace('_', ' ') for s, has in simulated_disease['symptoms'].items() if has])}]")
        
        # Start with a random symptom
        primary_symptoms = [s for s, has in simulated_disease["symptoms"].items() if has]
        if primary_symptoms:
            initial_symptom = random.choice(primary_symptoms)
            initial_message = f"I'm not feeling well. I have {initial_symptom.replace('_', ' ')}."
        else:
            initial_message = "I'm not feeling well."
        
        print(f"Patient: {initial_message}")
        conversation.append({"role": "user", "content": initial_message})
    else:
        print("Describe your symptoms, and I'll try to diagnose your condition.")
        
        # Get initial symptoms from user
        initial_message = input("Patient: ")
        conversation.append({"role": "user", "content": initial_message})
    
    turn_count = 0
    max_turns = 15  # Prevent infinite loops
    
    while turn_count < max_turns:
        turn_count += 1
        
        # Format conversation for model input
        formatted_conv = format_conversation(conversation)
        prompt = tokenizer.apply_chat_template(formatted_conv, tokenize=False, add_generation_prompt=True)
        
        # Generate doctor's response
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
        )
        
        response = model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=model.load_lora("grpodx_model"),
        )[0].outputs[0].text
        
        # Extract parts
        question_text = extract_question(response)
        reasoning_text = extract_reasoning(response)
        
        if not question_text:
            print("Doctor: I need more information to make a diagnosis.")
            question_text = "Can you tell me more about your symptoms?"
        
        # Display doctor's question
        print(f"Doctor: {question_text}")
        
        # Optionally show reasoning
        if reasoning_text and random.random() < 0.3:  # Show reasoning occasionally
            print(f"\n[Doctor's reasoning: {reasoning_text}]\n")
        
        # Add to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # Check if this is a final diagnosis
        diagnosis = extract_diagnosis(question_text)
        if diagnosis:
            # In simulation mode, verify the diagnosis
            if simulation_mode:
                if diagnosis.lower() == simulated_disease["disease_name"].lower():
                    print("\n[Correct diagnosis! ✓]")
                else:
                    print(f"\n[Incorrect diagnosis. The actual disease was: {simulated_disease['disease_name']}]")
            break
        
        # Get patient's response
        if simulation_mode:
            patient_reply = patient_response(question_text, simulated_disease)
            print(f"Patient: {patient_reply}")
        else:
            patient_reply = input("Patient: ")
            
            # Check for exit command
            if patient_reply.lower() in ["exit", "quit", "bye", "end"]:
                print("Ending diagnostic session.")
                break
        
        conversation.append({"role": "user", "content": patient_reply})
    
    print("\nDiagnostic session completed.")

# Main execution
if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_grpodx(num_steps=500)
    
    # Run interactive session
    interactive_diagnosis(model, tokenizer)