#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRPODx: Autonomous Diagnostic Agent with Reinforcement Learning
Built with Unsloth GRPO Framework

This implements a diagnostic agent that:
1. Interacts with a synthetic patient in a multi-turn dialogue
2. Dynamically asks relevant questions
3. Proposes a final diagnosis
4. Uses GRPO to reward correct diagnoses and improve diagnostic accuracy
"""

import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

# Import unsloth first as recommended
from unsloth import FastLanguageModel
import openai
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from vllm import SamplingParams

# Allow for direct supervised finetuning as fallback
try:
    from grpo_gemma_patch import patch_grpo_for_gemma, ensure_eager_attention
    # Apply the patch to fix Gemma matrix dimension issues in GRPO
    patch_grpo_for_gemma()
    print("Applied GRPO Gemma-specific patches")
except ImportError:
    print("GRPO Gemma patch not found - you may encounter tensor dimension errors")
    ensure_eager_attention = lambda x: x  # No-op function

# ======================================================================
# Configuration
# ======================================================================

# Model settings
MAX_SEQ_LENGTH = 1024  # Reduced context window for better compatibility
LORA_RANK = 8  # Reduced LoRA rank for lighter matrix operations
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"  # Base model
LOAD_IN_4BIT = True  # 4-bit quantization to fit in limited VRAM

# Training settings - reduced for Gemma compatibility
MAX_STEPS = 300  # Number of training steps
BATCH_SIZE = 1  # Batch size per device (following notebook)
GRAD_ACCUMULATION = 1  # Gradient accumulation steps
NUM_GENERATIONS = 4  # Reduced generation count for smaller tensors

# Paths
OUTPUT_DIR = "outputs"
SAVED_MODEL_PATH = "grpodx_model"  # Path to save the final model

# ======================================================================
# Dynamic Disease Generation and Simulation Environment
# ======================================================================


def generate_disease() -> Dict:
    """Generate a random disease with symptoms using GPT-4o-mini.
    
    Returns:
        Dictionary containing disease name and symptoms
    """
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
        """Initialize the patient agent.

        Args:
            disease_info: Disease definition with symptoms
            use_llm: Whether to use GPT-4o-mini for responses
        """
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
        """Remove XML tags and internal reasoning from doctor's message.
        
        This ensures the patient only sees and responds to the actual questions
        directed at them, not the doctor's internal reasoning.
        
        Args:
            message: Raw message from the doctor
            
        Returns:
            Cleaned message without XML sections
        """
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
        """Answer a question from the doctor based on symptoms.

        Args:
            question: Question from the doctor agent

        Returns:
            Patient's response based on symptoms
        """
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
        """Fallback rule-based response method.

        Args:
            question: Question from the doctor agent

        Returns:
            Rule-based patient response
        """
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


def run_episode(
    doctor_model,
    tokenizer,
    lora_adapter,
    disease_info: Dict,
    max_turns: int = 5,
    use_gpt_patient: bool = True,
) -> Tuple[List[Dict], str, float]:
    """Run a single diagnostic conversation episode.

    Args:
        doctor_model: The LLM model acting as doctor
        tokenizer: Tokenizer for the model
        lora_adapter: LoRA adapter for the model
        disease_info: Disease definition for the patient
        max_turns: Maximum number of conversation turns
        use_gpt_patient: Whether to use GPT-4o-mini for patient responses

    Returns:
        Tuple of (conversation history, final diagnosis, reward)
    """
    # Set up patient agent with the disease
    patient = PatientAgent(disease_info, use_llm=use_gpt_patient)

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
        }
    ]

    # Initial prompt from patient
    patient_initial = "Doctor, I'm not feeling well today."
    conversation.append({"role": "user", "content": patient_initial})

    # Main conversation loop
    final_diagnosis = None
    for turn in range(max_turns):
        # Get doctor's response
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        # Set up generation parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
        )

        # Generate doctor's response
        response = (
            doctor_model.fast_generate(
                prompt, sampling_params=sampling_params, lora_request=lora_adapter
            )[0]
            .outputs[0]
            .text
        )

        # Add doctor's response to conversation
        conversation.append({"role": "assistant", "content": response})

        # Check if final diagnosis was given
        if "final diagnosis:" in response.lower():
            # Extract the diagnosis
            diagnosis_match = re.search(
                r"final diagnosis:\s*([^.\n]+)", response.lower()
            )
            if diagnosis_match:
                final_diagnosis = diagnosis_match.group(1).strip()
            else:
                # Try to extract from the <answer> tag
                answer_match = re.search(r"<answer>\s*([^<]+)", response)
                if answer_match:
                    final_diagnosis = answer_match.group(1).strip()
            break

        # Get patient's response
        patient_response = patient.answer_question(response)
        conversation.append({"role": "user", "content": patient_response})

    # If no final diagnosis was given, force one in the last turn
    if final_diagnosis is None:
        prompt = tokenizer.apply_chat_template(
            conversation
            + [{"role": "user", "content": "Please provide your final diagnosis now."}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate final response with diagnosis
        sampling_params = SamplingParams(
            temperature=0.5,  # Lower temperature for more focused response
            top_p=0.95,
            max_tokens=256,
        )

        final_response = (
            doctor_model.fast_generate(
                prompt, sampling_params=sampling_params, lora_request=lora_adapter
            )[0]
            .outputs[0]
            .text
        )

        conversation.append(
            {"role": "user", "content": "Please provide your final diagnosis now."}
        )
        conversation.append({"role": "assistant", "content": final_response})

        # Extract the diagnosis
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

    # Calculate reward based on diagnostic accuracy
    reward = calculate_reward(final_diagnosis, disease_info)

    return conversation, final_diagnosis, reward


def calculate_reward(diagnosis: str, disease_info: Dict) -> float:
    """Calculate reward based on diagnostic accuracy.

    Args:
        diagnosis: The doctor's final diagnosis
        disease_info: The actual disease definition

    Returns:
        Reward value (1.0 for correct, partial for close match, 0 for incorrect)
    """
    # Get the correct disease name
    correct_disease = disease_info["disease_name"].lower()
    diagnosis = diagnosis.lower()

    # Exact match gets full reward
    if correct_disease in diagnosis:
        return 1.0

    # Check for partial matches or synonyms
    # This could be expanded with a medical knowledge base of related terms
    disease_terms = correct_disease.split()
    matched_terms = sum(1 for term in disease_terms if term in diagnosis)

    if matched_terms > 0:
        return 0.5 * (matched_terms / len(disease_terms))

    # No match
    return 0.0


# ======================================================================
# Reward Functions for GRPO
# ======================================================================


def calculate_numerical_reward(
    diagnosis: str, disease_info: Dict, reasoning: str = ""
) -> float:
    """Calculate a numerical reward score (0 to 1) using GPT-4o-mini.

    Args:
        diagnosis: The doctor's final diagnosis
        disease_info: The actual disease definition
        reasoning: The reasoning behind the diagnosis

    Returns:
        Numerical reward value between 0 and 1
    """
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


def grpo_reward_function(prompts, completions, disease_info, **kwargs) -> List[float]:
    """GRPO reward function that calculates numerical rewards for each completion.

    Args:
        prompts: Input prompts
        completions: Model completions
        disease_info: Actual disease information

    Returns:
        List of reward values between 0 and 1
    """
    rewards = []

    for completion in completions:
        content = completion[0]["content"]

        # Extract the diagnosis from completion
        diagnosis = None
        diagnosis_match = re.search(r"final diagnosis:\s*([^.\n]+)", content.lower())
        if diagnosis_match:
            diagnosis = diagnosis_match.group(1).strip()
        else:
            # Try to extract from the <answer> tag
            answer_match = re.search(r"<answer>\s*([^<]+)", content)
            if answer_match:
                diagnosis = answer_match.group(1).strip()

        # Extract reasoning if present
        reasoning = ""
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # If no diagnosis found, assign zero reward
        if not diagnosis:
            rewards.append(0.0)
            continue

        # Calculate the reward using GPT-4o-mini
        reward = calculate_numerical_reward(diagnosis, disease_info[0], reasoning)
        rewards.append(reward)

    return rewards


def format_reward_function(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion follows the required format.

    Args:
        completions: Model completions

    Returns:
        List of reward values
    """
    rewards = []

    for completion in completions:
        content = completion[0]["content"]
        reward = 0.0

        # Check for reasoning section with proper XML tags
        if "<reasoning>" in content and "</reasoning>" in content:
            reward += 0.5

            # Check reasoning quality by length (simple heuristic)
            reasoning = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
            if reasoning:
                reasoning_text = reasoning.group(1).strip()
                if (
                    len(reasoning_text.split()) >= 30
                ):  # Good reasoning has at least 30 words
                    reward += 0.25

        # Check for answer section with proper XML tags
        if "<answer>" in content and "</answer>" in content:
            reward += 0.25

        rewards.append(reward)

    return rewards


# ======================================================================
# Live Chat Interface
# ======================================================================


def run_simulated_chat(model, tokenizer, lora_adapter):
    """Run a simulated chat between AI doctor and GPT-4o-mini patient.
    
    This allows the user to observe a conversation between the doctor model
    and a simulated patient with a randomly generated disease.
    
    Args:
        model: The trained doctor model
        tokenizer: Tokenizer for the model
        lora_adapter: LoRA adapter for the model
    """
    print("\n" + "="*60)
    print("         SIMULATED DIAGNOSIS CHAT")
    print("="*60 + "\n")
    print("You will observe a conversation between the AI doctor and a simulated patient.")
    print("The patient will have a randomly generated disease.")
    print("You will see both the doctor's full response (including internal reasoning)")
    print("and what the patient actually sees after filtering out the reasoning.\n")
    
    # Generate a random disease
    disease = generate_disease()
    print("-"*60)
    print(f"GROUND TRUTH DISEASE: {disease['disease_name']}")
    print(f"Description: {disease.get('description', 'No description available')}")
    print(f"Symptoms present: {', '.join([s for s, v in disease['symptoms'].items() if v])}")
    print(f"Symptoms absent: {', '.join([s for s, v in disease['symptoms'].items() if not v])}")
    print("-"*60)
    print("\n--- Beginning of conversation ---\n")
    
    # Create patient agent
    patient = PatientAgent(disease, use_llm=True)
    
    # Initialize conversation with system prompt
    conversation = [
        {"role": "system", "content": """You are an expert medical diagnostician. 
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
</answer>"""}
    ]
    
    # Initial prompt from patient
    patient_initial = "Doctor, I'm not feeling well today."
    print(f"Patient's Initial Message:")
    print(f"{patient_initial}")
    conversation.append({"role": "user", "content": patient_initial})
    
    # Main conversation loop
    max_turns = 10
    for turn in range(max_turns):
        print(f"\n{'='*30} TURN {turn+1} {'='*30}")
        # Get doctor's response
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
        )
        
        response = model.fast_generate(
            prompt, 
            sampling_params=sampling_params,
            lora_request=lora_adapter
        )[0].outputs[0].text
        
        # Add doctor's response to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # Display doctor's full response (with reasoning) for the user
        print(f"Doctor's Full Response (including internal reasoning):")
        print(f"{response}")
        
        # Show what the patient actually sees (cleaned message)
        clean_response = patient._clean_doctor_message(response)
        print(f"\nWhat the patient actually sees:")
        print(f"{clean_response}")
        
        # Check if final diagnosis was given
        if "final diagnosis:" in response.lower():
            print("\nDiagnosis session complete.")
            
            # Extract diagnosis and calculate reward
            diagnosis_match = re.search(r"final diagnosis:\s*([^.\n]+)", response.lower())
            if diagnosis_match:
                final_diagnosis = diagnosis_match.group(1).strip()
            else:
                # Try to extract from the <answer> tag
                answer_match = re.search(r"<answer>\s*([^<]+)", response)
                if answer_match:
                    final_diagnosis = answer_match.group(1).strip()
                else:
                    final_diagnosis = "No clear diagnosis provided"
            
            # Calculate reward
            reward = calculate_reward(final_diagnosis, disease)
            
            # Display evaluation
            print(f"\nActual disease: {disease['disease_name']}")
            print(f"Doctor's diagnosis: {final_diagnosis}")
            print(f"Diagnostic accuracy: {reward:.2f}/1.00")
            break
        
        # Get patient's response
        patient_response = patient.answer_question(response)
        print(f"\nPatient's Response:")
        print(f"{patient_response}")
        conversation.append({"role": "user", "content": patient_response})
        
        # Ask user if they want to continue or exit
        user_input = input("\nPress Enter to continue or type 'exit' to end: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Session ended.")
            break
    
    print("\n--- End of conversation ---\n")


def run_interactive_chat(model, tokenizer, lora_adapter):
    """Run an interactive chat session with a real user as the patient.

    Args:
        model: The trained doctor model
        tokenizer: Tokenizer for the model
        lora_adapter: LoRA adapter for the model
    """
    print("\n========= INTERACTIVE DIAGNOSIS CHAT =========\n")
    print("You are now chatting with the AI doctor.")
    print("Describe your symptoms, and the AI will attempt to diagnose your condition.")
    print("Type 'quit' or 'exit' to end the session.\n")

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
        }
    ]

    # Get initial symptoms from user
    print(
        "Doctor: Hello, I'm your AI doctor. What symptoms are you experiencing today?"
    )

    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Session ended.")
        return

    conversation.append({"role": "user", "content": user_input})

    # Main conversation loop
    while True:
        # Generate doctor's response
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
        )

        response = (
            model.fast_generate(
                prompt, sampling_params=sampling_params, lora_request=lora_adapter
            )[0]
            .outputs[0]
            .text
        )

        # Add doctor's response to conversation
        conversation.append({"role": "assistant", "content": response})

        # Display doctor's response
        print(f"Doctor: {response}")

        # Check if final diagnosis was given
        if "final diagnosis:" in response.lower():
            print("\nDiagnosis session complete. Thank you for using the AI doctor.")
            break

        # Get user's response
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Session ended.")
            break

        conversation.append({"role": "user", "content": user_input})


# ======================================================================
# Dataset and Training Pipeline
# ======================================================================


def create_training_dataset(num_samples: int = 100) -> Dataset:
    """Create a dataset for training the diagnostic agent.

    Args:
        num_samples: Number of training samples to generate

    Returns:
        Dataset for GRPO training
    """
    data = []

    for _ in range(num_samples):
        # Generate a random disease using GPT-4o-mini
        disease = generate_disease()

        # Create a simple initial prompt
        prompt = [
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

        # Add to dataset
        data.append({"prompt": prompt, "disease_info": disease})

    return Dataset.from_list(data)


class OnlineGRPOTrainer:
    """Online GRPO Trainer that generates disease scenarios on-the-fly during training."""
    
    def __init__(self, model, tokenizer, lora_adapter=None, max_steps=300, batch_size=8, 
                 num_generations=8, max_turns=5, use_gpt_patient=True, output_dir="outputs"):
        """Initialize the online GRPO trainer.
        
        Args:
            model: The base model to train
            tokenizer: The tokenizer for the model
            lora_adapter: Optional LoRA adapter (for inference)
            max_steps: Maximum number of training steps
            batch_size: Batch size (must be divisible by num_generations)
            num_generations: Number of completions to generate per prompt
            max_turns: Maximum number of turns in doctor-patient conversation
            use_gpt_patient: Whether to use GPT-4o-mini for patient responses
            output_dir: Directory to save model checkpoints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.lora_adapter = lora_adapter
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_generations = num_generations
        self.max_turns = max_turns
        self.use_gpt_patient = use_gpt_patient
        self.output_dir = output_dir
        
        # Verify batch size is divisible by num_generations
        if self.batch_size % self.num_generations != 0:
            print(f"Warning: batch_size ({self.batch_size}) should be divisible by num_generations ({self.num_generations})")
            self.batch_size = self.num_generations
            print(f"Setting batch_size to {self.batch_size}")
        
        # Gemma-optimized GRPO configuration
        max_prompt_length = 128  # Even shorter prompt length for Gemma compatibility
        
        self.training_args = GRPOConfig(
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            num_generations=self.num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=MAX_SEQ_LENGTH - max_prompt_length,
            max_steps=1,  # Train for 1 step at a time for online learning
            save_steps=50,
            max_grad_norm=0.1,
            report_to="none",
            output_dir=self.output_dir,
            fp16=False,  # Avoid mixed precision to reduce complexity
            bf16=False,
        )
        
        # Track progress
        self.current_step = 0
        self.total_reward = 0.0
        self.best_reward = 0.0
        
    def _generate_episode_data(self):
        """Generate a new disease scenario and initial prompt for training."""
        # Generate a new random disease
        disease = generate_disease()
        print("\n" + "="*60)
        print(f"GENERATED DISEASE: {disease['disease_name']}")
        print(f"Description: {disease.get('description', 'No description')}")
        print(f"Symptoms present: {', '.join([s for s, v in disease['symptoms'].items() if v])}")
        print(f"Symptoms absent: {', '.join([s for s, v in disease['symptoms'].items() if not v])}")
        print("="*60 + "\n")
        
        # Create initial prompt
        prompt = [
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
        
        return {"prompt": prompt, "disease_info": disease}
        
    def _display_conversation_turn(self, doctor_msg, patient_msg, turn_num):
        """Display a single turn of doctor-patient conversation."""
        print(f"\n{'='*20} TURN {turn_num} {'='*20}")
        
        # Display doctor's message
        print("Doctor: " + doctor_msg)
        
        # Display patient's response if provided
        if patient_msg:
            print("\nPatient: " + patient_msg)
    
    def train(self, save_path=None):
        """Run online GRPO training, generating scenarios on-the-fly."""
        print("Starting online GRPO training...")
        
        while self.current_step < self.max_steps:
            # Generate new episode data
            print(f"\nStep {self.current_step + 1}/{self.max_steps}")
            print("Generating new disease scenario...")
            episode_data = self._generate_episode_data()
            
            # Create a one-example dataset
            train_dataset = Dataset.from_list([episode_data])
            
            # Create a GRPO trainer for this step
            trainer = GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                reward_funcs=[
                    grpo_reward_function,  # Primary diagnostic correctness reward
                    format_reward_function,  # Format adherence reward
                ],
                args=self.training_args,
                train_dataset=train_dataset,
            )
            
            # Run a single training step
            print("Running training step...")
            train_output = trainer.train()
            
            # Extract and display the average reward for this step
            step_reward = train_output.metrics.get("train_reward", 0.0)
            self.total_reward += step_reward
            avg_reward = self.total_reward / (self.current_step + 1)
            
            print(f"Step reward: {step_reward:.4f}, Average reward: {avg_reward:.4f}")
            
            # Simulate and show a doctor-patient conversation with current model
            if (self.current_step + 1) % 10 == 0 or self.current_step == 0:
                print("\n" + "="*60)
                print("DEMONSTRATING CURRENT MODEL CAPABILITIES")
                print("="*60)
                
                # Run a full conversation with the current model
                disease_info = episode_data["disease_info"]
                patient = PatientAgent(disease_info, use_llm=self.use_gpt_patient)
                
                # Initialize conversation
                conversation = [
                    {"role": "system", "content": episode_data["prompt"][0]["content"]},
                    {"role": "user", "content": "Doctor, I'm not feeling well today."},
                ]
                
                # Show the initial patient statement
                print("\nPatient: Doctor, I'm not feeling well today.")
                
                # Run conversation for several turns
                final_diagnosis = None
                for turn in range(self.max_turns):
                    # Get doctor's response
                    prompt = self.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                    
                    # Generate doctor's response
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=256,
                    )
                    
                    response = (
                        self.model.fast_generate(
                            prompt, sampling_params=sampling_params, lora_request=self.lora_adapter
                        )[0]
                        .outputs[0]
                        .text
                    )
                    
                    # Add doctor's response to conversation
                    conversation.append({"role": "assistant", "content": response})
                    
                    # Display what the patient sees (cleaned message)
                    clean_response = patient._clean_doctor_message(response)
                    
                    self._display_conversation_turn(
                        doctor_msg=f"{response}",
                        patient_msg="",
                        turn_num=turn+1
                    )
                    
                    print("\nWhat the patient actually sees:")
                    print(f"{clean_response}")
                    
                    # Check if final diagnosis was given
                    if "final diagnosis:" in response.lower():
                        # Extract the diagnosis
                        diagnosis_match = re.search(
                            r"final diagnosis:\s*([^.\n]+)", response.lower()
                        )
                        if diagnosis_match:
                            final_diagnosis = diagnosis_match.group(1).strip()
                        else:
                            # Try to extract from the <answer> tag
                            answer_match = re.search(r"<answer>\s*([^<]+)", response)
                            if answer_match:
                                final_diagnosis = answer_match.group(1).strip()
                        break
                    
                    # Get patient's response
                    patient_response = patient.answer_question(response)
                    conversation.append({"role": "user", "content": patient_response})
                    
                    # Print patient's response
                    print(f"\nPatient's Response:")
                    print(f"{patient_response}")
                
                # Calculate reward for this demonstration
                if final_diagnosis:
                    reward = calculate_reward(final_diagnosis, disease_info)
                    print(f"\nFinal diagnosis: {final_diagnosis}")
                    print(f"Actual disease: {disease_info['disease_name']}")
                    print(f"Reward: {reward:.2f}/1.00")
                else:
                    print("\nNo final diagnosis provided")
            
            # Save model checkpoint
            if (self.current_step + 1) % 50 == 0 and save_path:
                checkpoint_path = f"{save_path}_step{self.current_step+1}"
                print(f"Saving checkpoint to {checkpoint_path}...")
                self.model.save_lora(checkpoint_path)
            
            # Increment step counter
            self.current_step += 1
        
        # Save final model
        if save_path:
            print(f"Saving final model to {save_path}...")
            self.model.save_lora(save_path)
        
        print("Online GRPO training complete!")
        print(f"Average reward: {self.total_reward / self.max_steps:.4f}")
        
        return self.model


def main():
    """Main training pipeline."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GRPODx: Diagnostic Agent Training")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument(
        "--interact", action="store_true", help="Run interactive chat mode (you as patient)"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Run simulated chat with GPT-4o-mini patient"
    )
    parser.add_argument(
        "--use-gpt-patient",
        action="store_true",
        help="Use GPT-4o-mini for patient simulation",
    )
    parser.add_argument("--online", action="store_true", default=True, help="Use online GRPO training")
    parser.add_argument("--offline", action="store_true", help="Use traditional batch GRPO training instead of online")
    parser.add_argument("--sft", action="store_true", help="Use direct supervised fine-tuning (SFT) without GRPO")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Number of training steps")
    parser.add_argument("--num-tests", type=int, default=5, help="Number of test cases")
    parser.add_argument(
        "--model-path",
        type=str,
        default=SAVED_MODEL_PATH,
        help="Path to load/save model",
    )

    args = parser.parse_args()

    # If no arguments provided, default to training
    if not (args.train or args.test or args.interact or args.simulate):
        args.train = True
        
    # Handle flags - offline overrides online
    if args.offline:
        args.online = False

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    # Apply eager attention check and load model with Gemma-specific patches
    patched_model_name = ensure_eager_attention(MODEL_NAME)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=patched_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,  # Adjust based on available VRAM
        attn_implementation="eager",  # Use eager implementation for Gemma as recommended
    )

    if args.train:
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
            use_gradient_checkpointing="unsloth",  # Enable for long context
            random_state=42,
        )

        if args.offline:
            # Use traditional batch GRPO training
            print("Using traditional batch training (offline mode)...")
            print("Creating training dataset...")
            dataset = create_training_dataset(num_samples=100)
            
            # Configure GRPO training
            print("Configuring GRPO trainer...")
            # Gemma-optimized GRPO configuration for batch training
            max_prompt_length = 128  # Even shorter prompt length for Gemma compatibility
            
            training_args = GRPOConfig(
                learning_rate=5e-6,
                adam_beta1=0.9,
                adam_beta2=0.99,
                weight_decay=0.1,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                optim="paged_adamw_8bit",
                logging_steps=1,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUMULATION,
                num_generations=NUM_GENERATIONS,
                max_prompt_length=max_prompt_length,
                max_completion_length=MAX_SEQ_LENGTH - max_prompt_length,
                max_steps=args.steps,
                save_steps=50,
                max_grad_norm=0.1,
                report_to="none",
                output_dir=OUTPUT_DIR,
                fp16=False,  # Avoid mixed precision to reduce complexity
                bf16=False,
            )

            # Create GRPO trainer
            print("Creating GRPO trainer...")
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[
                    grpo_reward_function,  # Primary diagnostic correctness reward
                    format_reward_function,  # Format adherence reward
                ],
                args=training_args,
                train_dataset=dataset,
            )

            # Run training
            print("Starting GRPO training...")
            trainer.train()

            # Save the final model
            print(f"Saving model to {args.model_path}...")
            model.save_lora(args.model_path)
        else:
            # Default: Use online GRPO training (generates scenarios on-the-fly)
            print("Using online GRPO training (default mode)...")
            trainer = OnlineGRPOTrainer(
                model=model,
                tokenizer=tokenizer,
                max_steps=args.steps,
                batch_size=BATCH_SIZE,
                num_generations=NUM_GENERATIONS,
                max_turns=5,
                use_gpt_patient=args.use_gpt_patient or True,
                output_dir=OUTPUT_DIR
            )
            
            # Run online training
            trainer.train(save_path=args.model_path)

        print("Training complete!")

    # Test the model if requested
    if args.test:
        print(f"Testing model from {args.model_path}...")
        test_model(
            model,
            tokenizer,
            num_tests=args.num_tests,
            use_gpt_patient=args.use_gpt_patient,
        )

    # Run interactive mode if requested
    if args.interact:
        print(f"Loading model from {args.model_path} for interactive chat...")
        lora_adapter = model.load_lora(args.model_path)
        run_interactive_chat(model, tokenizer, lora_adapter)
        
    # Run simulated chat mode if requested
    if args.simulate:
        print(f"Loading model from {args.model_path} for simulated chat...")
        lora_adapter = model.load_lora(args.model_path)
        run_simulated_chat(model, tokenizer, lora_adapter)


def test_model(model, tokenizer, num_tests=5, use_gpt_patient=True):
    """Test the trained model on some sample cases.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        num_tests: Number of test cases to run
        use_gpt_patient: Whether to use GPT-4o-mini for patient responses
    """
    print("\n========= TESTING MODEL =========\n")

    # Load the trained LoRA adapter
    lora_adapter = model.load_lora(SAVED_MODEL_PATH)

    total_reward = 0.0

    for i in range(num_tests):
        # Generate a random disease for testing
        disease = generate_disease()
        print(f"Test case {i + 1}: Patient has {disease['disease_name']}")

        # Run a diagnostic episode
        conversation, diagnosis, reward = run_episode(
            model, tokenizer, lora_adapter, disease, use_gpt_patient=use_gpt_patient
        )

        # Print results
        print(f"Final diagnosis: {diagnosis}")
        print(f"Reward: {reward:.2f}")
        print("-" * 50)

        total_reward += reward

    # Print overall performance
    print(f"\nAverage reward: {total_reward / num_tests:.2f}")
    print("\n========= TESTING COMPLETE =========\n")


if __name__ == "__main__":
    main()
