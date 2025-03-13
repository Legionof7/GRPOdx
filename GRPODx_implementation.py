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
def patient_response(question, disease_info, conversation_history=None):
    """
    Simulate a patient response based on the disease info and conversation history
    
    Args:
        question: The doctor's question
        disease_info: Dictionary containing disease information and symptoms
        conversation_history: Optional list of previous messages
        
    Returns:
        A simulated patient response
    """
    symptoms = disease_info["symptoms"]
    question_lower = question.lower()
    
    # Track previous responses to avoid repetition
    previous_responses = []
    if conversation_history:
        previous_responses = [
            m["content"] for m in conversation_history if m["role"] == "user"
        ]
    
    # Check for repeated question
    if any(question in prev_q for prev_q in previous_responses[-3:] if isinstance(prev_q, str)):
        return "I already answered that question. Can you ask me something else?"
    
    # Detect if doctor is asking about multiple symptoms at once
    detected_symptoms = []
    for symptom, has_symptom in symptoms.items():
        # Simple keyword matching for symptoms in the question
        symptom_text = symptom.replace("_", " ")
        if symptom_text in question_lower:
            detected_symptoms.append((symptom, has_symptom))
    
    # If multiple symptoms detected, respond to all of them
    if len(detected_symptoms) > 1:
        responses = []
        for symptom, has_symptom in detected_symptoms:
            symptom_text = symptom.replace("_", " ")
            if has_symptom:
                responses.append(f"Yes, I have {symptom_text}")
            else:
                responses.append(f"No, I don't have {symptom_text}")
        return "Let me answer each part. " + ". ".join(responses) + "."
    
    # If only one symptom detected, give a direct answer
    elif len(detected_symptoms) == 1:
        symptom, has_symptom = detected_symptoms[0]
        symptom_text = symptom.replace("_", " ")
        if has_symptom:
            return f"Yes, I have {symptom_text}."
        else:
            return f"No, I don't have {symptom_text}."
    
    # Handle yes/no questions about general health status
    elif "how are you feeling" in question_lower or "how do you feel" in question_lower:
        # Count positive symptoms to determine how bad the patient feels
        positive_symptoms = sum(1 for _, has_symptom in symptoms.items() if has_symptom)
        if positive_symptoms > 3:
            return "I'm feeling really terrible right now."
        elif positive_symptoms > 1:
            return "I'm not feeling well at all."
        else:
            return "I'm feeling a bit under the weather, but not terrible."
    
    # Handle duration questions
    elif "how long" in question_lower or "when did" in question_lower or "started" in question_lower:
        return "These symptoms started about two days ago."
    
    # Handle severity questions
    elif "how severe" in question_lower or "how bad" in question_lower or "intensity" in question_lower:
        return "The symptoms are moderate, but they're affecting my daily activities."
    
    # Handle medication questions
    elif "medication" in question_lower or "medicine" in question_lower or "taking" in question_lower:
        return "I haven't taken any medication for this yet."
    
    # Handle medical history questions
    elif "history" in question_lower or "previous" in question_lower or "before" in question_lower:
        return "I've never experienced these exact symptoms before."
    
    # Default response for unrecognized questions
    return "I'm not sure about that. Can you ask me something more specific about my symptoms?"

# Extract parts from model output
def extract_reasoning(text):
    """Extract the reasoning from text, with safety checks"""
    # Safety check for None or non-string values
    if text is None:
        return None
    
    # Ensure text is a string
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None
    
    # Search for reasoning pattern
    try:
        reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
        match = re.search(reasoning_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Error extracting reasoning: {e}")
        
    return None

def extract_question(text):
    """Extract the question from text, with safety checks"""
    # Safety check for None or non-string values
    if text is None:
        return None
    
    # Ensure text is a string
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None
    
    # Search for question pattern in XML format
    try:
        question_pattern = r"<question>(.*?)</question>"
        match = re.search(question_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Error extracting question: {e}")
    
    # If no XML tags found, try to return the text itself if it looks like a question
    if text and "?" in text:
        return text
    
    return None

def extract_diagnosis(text):
    """Extract the diagnosis from text, with safety checks"""
    # Safety check for None or non-string values
    if text is None:
        return None
    
    # Ensure text is a string
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None
    
    # Search for diagnosis pattern
    try:
        diagnosis_pattern = r"Final diagnosis: ([A-Za-z\s\-]+)"
        match = re.search(diagnosis_pattern, text)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Error extracting diagnosis: {e}")
    
    # Alternative pattern in case the format is different
    try:
        alt_pattern = r"diagnosis:?\s*([A-Za-z\s\-]+)"
        match = re.search(alt_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except:
        pass
    
    return None

# Import LLM-based patient simulator
from llm_patient import llm_patient_response, create_llm_patient

# Episode simulation
def run_episode(model, tokenizer, disease_info=None, max_turns=10, use_llm_patient=False):
    """
    Run a complete diagnostic episode
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        disease_info: Optional disease info. If None, a new disease will be generated
        max_turns: Maximum number of conversation turns
        use_llm_patient: Whether to use the LLM-based patient simulator
        
    Returns:
        conversation, final_diagnosis, reward, disease_info
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
        
        # Generate patient's response - either using LLM or rule-based
        if use_llm_patient:
            patient_reply = llm_patient_response(
                question_text, 
                disease_info, 
                conversation, 
                model, 
                tokenizer
            )
        else:
            patient_reply = patient_response(question_text, disease_info, conversation)
            
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
def generate_training_batch(model, tokenizer, batch_size=4, completions_per_scenario=6, verbose=True, use_llm_patient=False):
    """
    Generate a batch of training data for GRPO using dynamically generated diseases
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        batch_size: Number of different disease scenarios to generate
        completions_per_scenario: Number of completions per disease scenario for GRPO
        verbose: Whether to print the generated conversations
        use_llm_patient: Whether to use the LLM-based patient simulator
    
    Returns:
        Dataset object containing training data
    """
    training_data = []
    disease_cache = []  # Cache of generated diseases for potential reuse
    
    for batch_idx in range(batch_size):
        # Generate a new disease scenario with 80% probability, or use cached one with 20% probability
        disease_scenario = get_disease(cache=disease_cache, use_cache_probability=0.2)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"TRAINING BATCH {batch_idx+1}/{batch_size} - DISEASE: {disease_scenario['disease_name']}")
            print(f"SYMPTOMS: {', '.join([s.replace('_', ' ') for s, has in disease_scenario['symptoms'].items() if has])}")
            print(f"{'='*80}")
        
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
            
            if verbose:
                print(f"GENERATED RELATED DISEASE: {disease_scenario['disease_name']}")
                print(f"SYMPTOMS: {', '.join([s.replace('_', ' ') for s, has in disease_scenario['symptoms'].items() if has])}")
        
        episode_data = []
        for episode_idx in range(completions_per_scenario):
            # Run an episode with the same disease
            conversation, diagnosis, reward, _ = run_episode(
                model, 
                tokenizer, 
                disease_scenario,
                use_llm_patient=use_llm_patient
            )
            
            if verbose:
                print(f"\n--- EPISODE {episode_idx+1}/{completions_per_scenario} ---")
                print(f"Generated conversation:")
                for turn_idx, message in enumerate(conversation):
                    role = message["role"]
                    content = message["content"]
                    if role == "user":
                        print(f"Patient: {content}")
                    elif role == "assistant":
                        print(f"Doctor: {content}")
                    else:
                        print(f"{role.capitalize()}: {content}")
                
                if diagnosis:
                    print(f"\nFinal diagnosis: {diagnosis}")
                    print(f"Correct diagnosis: {disease_scenario['disease_name']}")
                    print(f"Reward: {reward}")
                else:
                    print("\nNo final diagnosis provided")
                    print(f"Reward: {reward}")
                print("---\n")
            
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

# Custom callback function to monitor training progress
from transformers.trainer_callback import TrainerCallback

class TrainingCallback(TrainerCallback):
    def __init__(self, model, tokenizer, print_frequency=10, use_llm_patient=False):
        self.model = model
        self.tokenizer = tokenizer
        self.print_frequency = print_frequency
        self.step = 0
        self.example_scenarios = []
        self.last_rewards = []
        self.use_llm_patient = use_llm_patient
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of trainer initialization"""
        return control
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if state.log_history and len(state.log_history) > 0:
            latest_log = state.log_history[-1]
            reward = latest_log.get('reward', 0)
            self.last_rewards.append(reward)
            
            # Keep only the last 20 rewards
            if len(self.last_rewards) > 20:
                self.last_rewards = self.last_rewards[-20:]
            
            # Calculate average reward over recent steps
            avg_reward = sum(self.last_rewards) / len(self.last_rewards)
            
            if self.step % self.print_frequency == 0:
                # Generate and print a test conversation at regular intervals
                print(f"\n{'='*40} TRAINING PROGRESS {'='*40}")
                print(f"Step: {self.step}/{args.max_steps} | Average Reward: {avg_reward:.4f}")
                
                # Generate a test conversation with a random disease
                if len(self.example_scenarios) < 5:
                    # Generate a few diseases to use for consistent testing
                    disease = generate_random_disease()
                    self.example_scenarios.append(disease)
                else:
                    # Rotate through the example scenarios
                    disease = self.example_scenarios[self.step % len(self.example_scenarios)]
                
                # Print disease info
                print(f"\nTEST DISEASE: {disease['disease_name']}")
                symptoms = [s.replace('_', ' ') for s, has in disease['symptoms'].items() if has]
                print(f"SYMPTOMS: {', '.join(symptoms)}")
                
                # Generate a conversation
                print("\nGENERATING TEST CONVERSATION...")
                try:
                    conversation, diagnosis, reward, _ = run_episode(
                        self.model, self.tokenizer, disease, max_turns=8,
                        use_llm_patient=self.use_llm_patient
                    )
                    
                    # Print the conversation
                    for message in conversation:
                        role = message["role"]
                        content = message["content"]
                        if role == "user":
                            print(f"Patient: {content}")
                        elif role == "assistant":
                            print(f"Doctor: {content}")
                        else:
                            print(f"{role.capitalize()}: {content}")
                    
                    # Print the diagnosis and reward
                    if diagnosis:
                        print(f"\nFinal diagnosis: {diagnosis}")
                        print(f"Correct diagnosis: {disease['disease_name']}")
                        print(f"Reward: {reward}")
                    else:
                        print("\nNo final diagnosis provided")
                        print(f"Reward: {reward}")
                except Exception as e:
                    print(f"Error generating test conversation: {e}")
                
                print(f"{'='*90}\n")
        
        return control

# Main training function
def train_grpodx(num_steps=500, batch_size=4, completions_per_scenario=6, verbose=True, use_llm_patient=False):
    """
    Main training function for GRPODx
    
    Args:
        num_steps: Number of training steps
        batch_size: Number of different diseases per batch
        completions_per_scenario: Number of completions per disease for GRPO
        verbose: Whether to print detailed logs during training
        use_llm_patient: Whether to use the LLM-based patient simulator
    """
    # Set up model parameters
    max_seq_length = 4096  # Increased from 2048 to allow longer conversations
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
    max_prompt_length = 1024  # Increased from 512 to allow more context in prompts
    
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
    initial_dataset = generate_training_batch(
        model, 
        tokenizer, 
        batch_size, 
        completions_per_scenario, 
        verbose=verbose,
        use_llm_patient=use_llm_patient
    )
    
    # Create custom reward function
    def diagnosis_reward_func(prompts, completions, **kwargs):
        """Custom reward function for GRPODx"""
        rewards = []
        
        # Extract prompt information to get the correct disease for each scenario
        scenario_diseases = []
        for prompt in prompts:
            # Try to extract the disease name from the prompt (context)
            disease_name = None
            try:
                # First check if we have a direct reference to the disease in the context
                if isinstance(prompt, dict) and "context" in prompt:
                    # Try to extract disease name from the context
                    match = re.search(r"disease is ([A-Za-z\s\-]+)", prompt["context"], re.IGNORECASE)
                    if match:
                        disease_name = match.group(1).strip()
                # If we can't find a direct reference, look at cached diseases
                if not disease_name and len(disease_cache) > 0:
                    # Use the most recently added disease as a fallback
                    disease_name = disease_cache[-1].get("disease_name", "")
            except Exception as e:
                print(f"Warning: Error extracting disease from prompt: {e}")
            
            scenario_diseases.append(disease_name)
            
        # Extract all known disease names as a fallback
        all_disease_names = []
        try:
            # First try to get disease names from the training dataset
            if hasattr(initial_dataset, 'features') and 'disease_name' in initial_dataset.features:
                all_disease_names = list(set(initial_dataset['disease_name']))
        except Exception as e:
            print(f"Warning: Could not extract disease names from dataset: {e}")
        
        # Add example disease names as fallback
        example_diseases = [d["disease_name"] for d in DISEASE_EXAMPLES]
        all_disease_names.extend(example_diseases)
            
        # Add any generated diseases
        if len(disease_cache) > 0:
            all_disease_names.extend([d.get("disease_name", "") for d in disease_cache])
        
        # Ensure all disease names are strings
        all_disease_names = [str(name) for name in all_disease_names if name]
        
        for i, completion in enumerate(completions):
            # Handle different completion formats with robust error handling
            try:
                # Handle different completion formats - could be a string or a dict
                if isinstance(completion, list) and len(completion) > 0:
                    if isinstance(completion[0], dict) and "content" in completion[0]:
                        response = completion[0]["content"]
                    else:
                        response = str(completion[0])
                elif isinstance(completion, dict):
                    if "content" in completion:
                        response = completion["content"]
                    elif "text" in completion:
                        response = completion["text"]
                    else:
                        response = str(completion)
                elif isinstance(completion, str):
                    response = completion
                else:
                    response = str(completion)
            except Exception as e:
                print(f"Warning: Error parsing completion: {e}")
                try:
                    response = str(completion)
                except:
                    response = "Error parsing completion"
            
            # Get diagnosis from response
            diagnosis = extract_diagnosis(extract_question(response))
            
            # Initialize reward
            reward = 0.0
            
            # If no diagnosis, no base reward (discourages non-diagnostic responses)
            if not diagnosis:
                # The model should only get 0 reward here, not a small positive reward
                reward = 0.0
            else:
                # Base reward for providing any diagnosis in correct format
                reward = 0.3
                
                # Try to match with the correct disease for this scenario
                correct_disease = scenario_diseases[i % len(scenario_diseases)] if scenario_diseases else None
                
                if correct_disease and correct_disease.lower() == diagnosis.lower():
                    # Perfect match with correct disease - highest reward
                    reward = 1.0
                elif correct_disease and correct_disease.lower() in diagnosis.lower():
                    # Partial match with correct disease
                    reward = 0.8
                else:
                    # Check if it matches any known disease name (less valuable than correct match)
                    for disease_name in all_disease_names:
                        if disease_name.lower() in diagnosis.lower():
                            reward = 0.5  # Better than random text, but not the right disease
                            break
                
                # Bonus for good formatting with XML tags
                if "<reasoning>" in response and "</reasoning>" in response:
                    reward += 0.1
                
                if "<question>" in response and "</question>" in response:
                    reward += 0.1
                
                # Check for question quality
                questions_count = response.count("<question>")
                if questions_count > 1:
                    # Penalize for asking multiple questions at once
                    reward -= 0.1 * (questions_count - 1)
            
            rewards.append(reward)
        
        return rewards
    
    # Create a callback for monitoring
    if verbose:
        callback = TrainingCallback(model, tokenizer, print_frequency=20, use_llm_patient=use_llm_patient)
    else:
        callback = None
    
    # Setup GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[diagnosis_reward_func],
        args=training_args,
        train_dataset=initial_dataset,
        callbacks=[callback] if callback else None,
    )
    
    # Attach disease cache to trainer for reference (won't be used directly by GRPOTrainer)
    trainer.disease_cache = disease_cache
    
    if verbose:
        print(f"\n{'='*30} STARTING TRAINING {'='*30}")
        print(f"Model will train for {num_steps} steps")
        print(f"Batch size: {batch_size}, Completions per scenario: {completions_per_scenario}")
        print(f"Will generate a diagnostic conversation every 20 steps")
        print(f"{'='*70}\n")
    
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