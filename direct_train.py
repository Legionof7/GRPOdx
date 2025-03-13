#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct training approach for GRPODx
"""

import argparse
import json
import os
import random
import re
import torch
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AdamW, get_scheduler
from unsloth import FastLanguageModel
from vllm import SamplingParams

# ======================================================================
# Configuration
# ======================================================================

# Model settings
MAX_SEQ_LENGTH = 1024  # Reduced context window for better compatibility
LORA_RANK = 8  # Reduced LoRA rank for lighter matrix operations
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"  # Base model
LOAD_IN_4BIT = True  # 4-bit quantization to fit in limited VRAM

# Training settings
MAX_STEPS = 30  # Number of training steps for direct approach
LEARNING_RATE = 2e-5
BATCH_SIZE = 1
NUM_EPISODES = 3  # Episodes per step

# Paths
OUTPUT_DIR = "outputs"
SAVED_MODEL_PATH = "grpodx_model"

# ======================================================================
# OpenAI API Configuration
# ======================================================================

import openai

def generate_disease() -> Dict:
    """Generate a random disease with symptoms using GPT-4o-mini."""
    try:
        system_message = """You are a medical disease generator. 
Your task is to create a realistic disease scenario with associated symptoms.
Return ONLY a JSON object with the following format:
{
  "disease_name": "Disease Name",
  "symptoms": {
    "symptom1": true,  // Present symptoms are true
    "symptom2": true,
    "symptom3": false,  // Absent symptoms are false
    "symptom4": false
  },
  "description": "Brief description of the disease"
}

Include at least 8-10 symptoms (mix of present and absent).
Make the disease realistic but vary between common and rare conditions.
Do not include any explanatory text, ONLY return the JSON object."""

        user_message = "Generate a random disease with realistic symptoms and indicate which symptoms are present (true) or absent (false)."
        
        # Updated OpenAI API call for v1.0.0+
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        
        disease_json = response.choices[0].message.content
        
        # Clean up the response if needed (sometimes GPT adds markdown formatting)
        disease_json = disease_json.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON
        disease_info = json.loads(disease_json)
        
        # Ensure required fields exist
        if "disease_name" not in disease_info or "symptoms" not in disease_info:
            raise ValueError("Generated disease missing required fields")
            
        return disease_info
    
    except Exception as e:
        print(f"Error generating disease: {e}")
        # Fallback to a basic disease if generation fails
        return {
            "disease_name": "Influenza",
            "symptoms": {
                "fever": True,
                "cough": True,
                "headache": True,
                "sore_throat": True,
                "muscle_pain": True,
                "fatigue": True,
                "runny_nose": True,
                "sneezing": False,
                "chest_pain": False,
                "shortness_of_breath": False
            },
            "description": "A common viral infection affecting the respiratory system"
        }

class PatientAgent:
    """Simulates a patient with a specific disease using GPT-4o-mini."""

    def __init__(self, disease_info: Dict, use_llm: bool = True):
        """Initialize the patient agent."""
        self.disease = disease_info
        self.symptoms = disease_info["symptoms"]
        self.name = disease_info["disease_name"]
        self.use_llm = use_llm

        # Initialize the chat history
        self.chat_history = [
            {
                "role": "system",
                "content": f"""You are a patient with {self.name}.
Your symptoms include: {', '.join([s for s, v in self.symptoms.items() if v])}
You DO NOT have these symptoms: {', '.join([s for s, v in self.symptoms.items() if not v])}

When the doctor asks about your symptoms, answer truthfully based on the symptoms list.
Be realistic and provide plausible answers a real patient might give.
Add appropriate details that would be consistent with your condition.
Do not directly state your diagnosis, but describe your symptoms clearly.
Keep your responses relatively brief (1-3 sentences).
""",
            }
        ]

        # Track which symptoms have been discussed
        self.asked_symptoms = set()

    def _clean_doctor_message(self, message: str) -> str:
        """Remove XML tags and internal reasoning from doctor's message."""
        # Remove <reasoning>...</reasoning> sections
        message = re.sub(r'<reasoning>.*?</reasoning>', '', message, flags=re.DOTALL)
        
        # Remove <answer>...</answer> sections
        message = re.sub(r'<answer>.*?</answer>', '', message, flags=re.DOTALL)
        
        # Clean up any "Final diagnosis:" sections
        message = re.sub(r'Final diagnosis:.*?(\n|$)', '', message, flags=re.DOTALL)
        
        # Remove any trailing whitespace or redundant newlines
        message = re.sub(r'\n{3,}', '\n\n', message.strip())
        
        if not message:
            # If nothing is left after cleaning, return a generic message
            return "Could you tell me about your symptoms?"
            
        return message
    
    def answer_question(self, question: str) -> str:
        """Answer a question from the doctor based on symptoms."""
        # Clean the doctor's message to remove reasoning/answer sections
        clean_question = self._clean_doctor_message(question)
        
        # Add the cleaned doctor's question to chat history
        self.chat_history.append({"role": "user", "content": clean_question})

        if self.use_llm:
            try:
                # Use GPT-4o-mini to generate a patient response
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self.chat_history,
                    temperature=0.7,
                    max_tokens=150,
                )

                patient_response = response.choices[0].message.content

                # Add the response to chat history
                self.chat_history.append(
                    {"role": "assistant", "content": patient_response}
                )

                return patient_response

            except Exception as e:
                print(f"Error using GPT-4o-mini: {e}")
                # Fall back to rule-based responses if API call fails
                return self._rule_based_response(question)
        else:
            # Use rule-based responses
            return self._rule_based_response(question)

    def _rule_based_response(self, question: str) -> str:
        """Fallback rule-based response method."""
        question_lower = question.lower()

        # Check for final diagnosis attempt
        if "final diagnosis" in question_lower:
            return "I'm waiting for your diagnosis."

        # Look for symptom mentions in the question
        for symptom in self.symptoms:
            if symptom in question_lower:
                self.asked_symptoms.add(symptom)
                if self.symptoms[symptom]:
                    return f"Yes, I do have {symptom}. It started a few days ago and has been bothering me."
                else:
                    return f"No, I don't have {symptom}."

        # Generic responses for general questions
        if (
            "how are you feeling" in question_lower
            or "what brings you" in question_lower
        ):
            # Return the top 2 symptoms if they exist
            present_symptoms = [s for s in self.symptoms if self.symptoms[s]][:2]
            if present_symptoms:
                symptoms_text = " and ".join(present_symptoms)
                return f"I'm not feeling well. I have {symptoms_text} for the past few days."
            return "I'm not feeling well, but it's hard to describe."

        if "how long" in question_lower or "when did" in question_lower:
            return (
                "The symptoms started about 3-4 days ago and have been getting worse."
            )

        if "medication" in question_lower:
            return "I've only taken some over-the-counter pain relievers, but they didn't help much."

        if "allergies" in question_lower:
            return "I don't have any known allergies to medications."

        if "medical history" in question_lower or "previous" in question_lower:
            return "I've been generally healthy. No major medical issues in the past."

        # Default response if no specific symptom was asked about
        return "Could you please ask more specific questions about my symptoms?"

def calculate_reward(diagnosis: str, disease_info: Dict) -> float:
    """Calculate reward based on diagnostic accuracy."""
    # Get the correct disease name
    correct_disease = disease_info["disease_name"].lower()
    diagnosis = diagnosis.lower()

    # Exact match gets full reward
    if correct_disease in diagnosis:
        return 1.0

    # Check for partial matches or synonyms
    disease_terms = correct_disease.split()
    matched_terms = sum(1 for term in disease_terms if term in diagnosis)

    if matched_terms > 0:
        return 0.5 * (matched_terms / len(disease_terms))

    # No match
    return 0.0

def calculate_numerical_reward(diagnosis: str, disease_info: Dict, reasoning: str = "") -> float:
    """Calculate a numerical reward score (0 to 1) using GPT-4o-mini."""
    # Get the ground truth disease and symptoms
    correct_disease = disease_info["disease_name"]

    symptoms_present = [s for s, v in disease_info["symptoms"].items() if v]
    symptoms_absent = [s for s, v in disease_info["symptoms"].items() if not v]

    # First, try using GPT-4o-mini to calculate the reward score
    try:
        # Construct the system message
        system_message = """You are a medical evaluation system. Your task is to evaluate a doctor's diagnosis 
given a set of symptoms and the correct diagnosis. Provide a score from 0.0 to 1.0, where:
- 1.0: Perfect diagnosis that matches the correct disease exactly
- 0.75-0.99: Very good diagnosis that identifies the correct disease with minor inaccuracies
- 0.5-0.74: Partial match - identifies some aspects of the disease or a related condition
- 0.25-0.49: Poor match but shows some understanding of symptoms
- 0.0-0.24: Completely incorrect diagnosis or failed to provide diagnosis

ONLY respond with a single number between 0.0 and 1.0 with one decimal place. Do not include any explanation."""

        # Create the user prompt
        user_prompt = f"""
Correct Diagnosis: {correct_disease}
Symptoms Present: {', '.join(symptoms_present)}
Symptoms Absent: {', '.join(symptoms_absent)}

Doctor's Diagnosis: {diagnosis}
Doctor's Reasoning: {reasoning}

Evaluate and provide a score from 0.0 to 1.0:"""

        # Get the evaluation from GPT-4o-mini
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=10,
        )

        # Extract the score from the response
        score_text = response.choices[0].message.content.strip()

        # Try to convert the score to a float
        try:
            score = float(score_text)
            # Ensure the score is between 0 and 1
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            # If we can't convert to float, fall back to heuristic scoring
            print(
                f"Could not convert score '{score_text}' to float. Using fallback scoring."
            )
    except Exception as e:
        print(f"GPT-4o-mini evaluation failed: {e}")

    # Fallback heuristic scoring if GPT-4o-mini fails
    diagnosis = diagnosis.lower()
    correct_disease = correct_disease.lower()

    # Exact match gets full reward
    if correct_disease in diagnosis:
        # Scale based on exactness of match
        if diagnosis.strip() == correct_disease.strip():
            return 1.0
        else:
            return 0.9  # Good but not exact

    # Check for partial matches
    disease_terms = correct_disease.split()
    matched_terms = sum(1 for term in disease_terms if term in diagnosis)

    if matched_terms > 0:
        # Scale based on how many terms match
        return 0.5 * (matched_terms / len(disease_terms))

    # Check if reasoning shows understanding of key symptoms
    if reasoning:
        reasoning = reasoning.lower()
        symptom_mentioned = sum(
            1 for symptom in symptoms_present if symptom in reasoning
        )
        if symptom_mentioned > len(symptoms_present) / 2:
            return 0.3  # Shows some understanding of symptoms

    # No match
    return 0.0

class DirectTrainer:
    def __init__(self, model, tokenizer, 
                 learning_rate=LEARNING_RATE, 
                 num_steps=MAX_STEPS, 
                 use_gpt_patient=True,
                 episodes_per_step=NUM_EPISODES):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.use_gpt_patient = use_gpt_patient
        self.episodes_per_step = episodes_per_step
        
        # Set up optimizer with just LoRA parameters
        self.optimizer = self._setup_optimizer()
        
        # Set up scheduler
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.1 * num_steps),
            num_training_steps=num_steps
        )
        
        # Track statistics
        self.total_reward = 0
        self.total_episodes = 0
        self.step = 0
        
    def _setup_optimizer(self):
        """Set up the optimizer with just the trainable (LoRA) parameters."""
        # Get only the trainable parameters (LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer
        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        return optimizer
        
    def run_episode(self, disease_info):
        """Run a single training episode."""
        # Create patient agent
        patient = PatientAgent(disease_info, use_llm=self.use_gpt_patient)
        
        # Initialize conversation with system prompt
        conversation = [
            {
                "role": "system",
                "content": """You are an expert medical diagnostician. 
Your goal is to determine the patient's condition by asking relevant questions.
After gathering sufficient information, provide your final diagnosis.

When you're ready to give your diagnosis, format it as:
Final diagnosis: [your diagnosis]

Format your reasoning as:
<reasoning>
Your step-by-step diagnostic reasoning here...
</reasoning>

Format your final answer as:
<answer>
Your final diagnosis here
</answer>""",
            },
            {"role": "user", "content": "Doctor, I'm not feeling well today."},
        ]
        
        # Training data
        inputs = []
        outputs = []
        
        # Keep track of the diagnosis and reasoning
        final_diagnosis = None
        reasoning = ""
        
        # Main conversation loop (up to 5 turns)
        for turn in range(5):
            # Create input prompt
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            
            # Add to inputs
            inputs.append(prompt)
            
            # Generate the model's response
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=256,
            )
            
            generated_text = (
                self.model.fast_generate(
                    prompt, sampling_params=sampling_params
                )[0]
                .outputs[0]
                .text
            )
            
            # Add to outputs
            outputs.append(generated_text)
            
            # Print conversation for visibility
            print(f"\n==== TURN {turn+1} ====")
            print(f"Doctor: {generated_text}")
            
            # Add the doctor's response to the conversation
            conversation.append({"role": "assistant", "content": generated_text})
            
            # Check if final diagnosis was given
            if "final diagnosis:" in generated_text.lower():
                # Try to extract the diagnosis
                diagnosis_match = re.search(
                    r"final diagnosis:\s*([^.\n]+)", generated_text.lower()
                )
                if diagnosis_match:
                    final_diagnosis = diagnosis_match.group(1).strip()
                else:
                    # Try to extract from the <answer> tag
                    answer_match = re.search(r"<answer>\s*([^<]+)", generated_text)
                    if answer_match:
                        final_diagnosis = answer_match.group(1).strip()
                
                # Extract reasoning if available
                reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", generated_text, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                
                break
            
            # Get patient's response
            patient_response = patient.answer_question(generated_text)
            print(f"Patient: {patient_response}")
            
            # Add the patient's response to the conversation
            conversation.append({"role": "user", "content": patient_response})
        
        # If no diagnosis was provided, ask for one explicitly
        if not final_diagnosis:
            # Add request for diagnosis
            conversation.append({"role": "user", "content": "Please provide your final diagnosis now."})
            
            # Create input prompt
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            
            # Add to inputs
            inputs.append(prompt)
            
            # Generate response with diagnosis
            sampling_params = SamplingParams(
                temperature=0.5,
                top_p=0.95,
                max_tokens=256,
            )
            
            final_response = (
                self.model.fast_generate(
                    prompt, sampling_params=sampling_params
                )[0]
                .outputs[0]
                .text
            )
            
            # Add to outputs
            outputs.append(final_response)
            
            print(f"\nDoctor (Final Response): {final_response}")
            
            # Try to extract the diagnosis
            diagnosis_match = re.search(
                r"final diagnosis:\s*([^.\n]+)", final_response.lower()
            )
            if diagnosis_match:
                final_diagnosis = diagnosis_match.group(1).strip()
            else:
                # Try to extract from the <answer> tag
                answer_match = re.search(r"<answer>\s*([^<]+)", final_response)
                if answer_match:
                    final_diagnosis = answer_match.group(1).strip()
                else:
                    final_diagnosis = "No clear diagnosis provided"
            
            # Extract reasoning if available
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", final_response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
        
        # Calculate reward
        reward = calculate_numerical_reward(final_diagnosis, disease_info, reasoning)
        
        # Print episode results
        print(f"\nDisease: {disease_info['disease_name']}")
        print(f"Final Diagnosis: {final_diagnosis}")
        print(f"Reward: {reward:.2f}")
        
        return inputs, outputs, reward, final_diagnosis
        
    def train_step(self):
        """Execute a single training step with multiple episodes."""
        self.model.train()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        step_reward = 0
        step_episodes = 0
        
        # Run multiple episodes per step
        for _ in range(self.episodes_per_step):
            # Generate a new disease
            disease_info = generate_disease()
            
            # Run episode
            inputs, outputs, reward, _ = self.run_episode(disease_info)
            
            # Only use episodes with positive reward for training
            if reward > 0:
                # Use the final exchange for training
                final_input = inputs[-1]
                final_output = outputs[-1]
                
                # Tokenize input and target
                inputs_tokens = self.tokenizer(
                    final_input, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH
                ).to(self.model.device)
                
                # For the target, prepare the labels for language modeling
                labels = self.tokenizer(
                    final_output,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH
                ).input_ids.to(self.model.device)
                
                # Set labels to -100 for input tokens (don't compute loss on them)
                input_length = len(self.tokenizer.encode(final_input)) - 1  # -1 for the EOS token
                labels[0, :input_length] = -100
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs_tokens.input_ids,
                    attention_mask=inputs_tokens.attention_mask,
                    labels=labels
                )
                
                # Scale loss by reward (higher reward = more emphasis on this example)
                loss = outputs.loss * (2.0 - reward)
                
                # Backward pass
                loss.backward()
                
                # Add to step statistics
                step_reward += reward
                step_episodes += 1
                
                # Report on this episode
                print(f"Episode Loss: {loss.item():.4f}, Reward: {reward:.2f}")
            else:
                print(f"Skipping training for episode with reward {reward:.2f}")
        
        # Update model parameters
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update statistics
        if step_episodes > 0:
            avg_step_reward = step_reward / step_episodes
        else:
            avg_step_reward = 0
            
        self.total_reward += step_reward
        self.total_episodes += step_episodes
        self.step += 1
        
        # Calculate overall statistics
        if self.total_episodes > 0:
            avg_reward = self.total_reward / self.total_episodes
        else:
            avg_reward = 0
            
        return avg_step_reward, avg_reward
        
    def train(self, save_path=None):
        """Train the model with online reinforcement learning."""
        print("Starting direct reinforcement learning training...")
        
        for step in range(self.num_steps):
            print(f"\n=========== STEP {step+1}/{self.num_steps} ===========")
            
            # Execute a training step
            avg_step_reward, avg_reward = self.train_step()
            
            # Report progress
            print(f"Step {step+1} completed.")
            print(f"Step Reward: {avg_step_reward:.4f}, Overall Reward: {avg_reward:.4f}")
            
            # Save checkpoint
            if save_path and (step + 1) % 10 == 0:
                checkpoint_path = f"{save_path}_step{step+1}"
                print(f"Saving checkpoint to {checkpoint_path}")
                self.model.save_lora(checkpoint_path)
        
        # Save final model
        if save_path:
            print(f"Saving final model to {save_path}")
            self.model.save_lora(save_path)
            
        print(f"Training complete! Final average reward: {avg_reward:.4f}")
        return self.model
        
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Direct RL Trainer for GRPODx")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Number of training steps")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Episodes per step")
    parser.add_argument("--use-gpt-patient", action="store_true", help="Use GPT-4o-mini for patient simulation")
    parser.add_argument("--model-path", type=str, default=SAVED_MODEL_PATH, help="Path to save model")
    
    args = parser.parse_args()
    
    # Set environment variable for eager attention
    os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
    
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,
        attn_implementation="eager",
    )
    
    # Initialize LoRA model for training
    print("Initializing LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Create the trainer
    trainer = DirectTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=LEARNING_RATE,
        num_steps=args.steps,
        use_gpt_patient=args.use_gpt_patient,
        episodes_per_step=args.episodes
    )
    
    # Run training
    trainer.train(save_path=args.model_path)
    
if __name__ == "__main__":
    main()