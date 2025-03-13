"""
LLM-based Patient Simulator

This module implements a patient simulator that uses a language model to generate
realistic patient responses based on a disease profile.
"""

import random
from typing import Dict, List, Any, Optional, Union

class LLMPatientSimulator:
    """
    A patient simulator that uses a language model to generate realistic
    patient responses based on a disease profile.
    """
    
    def __init__(self, model, tokenizer, disease_info=None):
        """
        Initialize the patient simulator
        
        Args:
            model: A language model (vLLM, Hugging Face, etc.)
            tokenizer: The tokenizer for the model
            disease_info: Optional disease information. If None, a random disease will be generated.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.disease_info = disease_info
        self.conversation_history = []
        self.patient_profile = {}
        
        # Generate a patient profile if needed
        if disease_info:
            self.initialize_profile(disease_info)
    
    def initialize_profile(self, disease_info):
        """
        Initialize the patient profile based on the disease information
        
        Args:
            disease_info: Disease information dictionary
        """
        disease_name = disease_info.get("disease_name", "Unknown Condition")
        symptoms = disease_info.get("symptoms", {})
        
        # Create a basic patient profile with age, gender, etc.
        self.patient_profile = {
            "disease": disease_name,
            "symptoms": symptoms,
            "age": random.randint(25, 75),
            "gender": random.choice(["male", "female"]),
            "medical_history": [],
            "current_medications": []
        }
        
        # Add some random medical history items (20% chance for each)
        possible_conditions = [
            "High blood pressure", "Diabetes", "Asthma", "Allergies", 
            "Heart disease", "Depression", "Anxiety", "Arthritis"
        ]
        
        for condition in possible_conditions:
            if random.random() < 0.2:
                self.patient_profile["medical_history"].append(condition)
        
        # Add some random medications (10% chance for each)
        possible_medications = [
            "Lisinopril", "Metformin", "Atorvastatin", "Levothyroxine", 
            "Albuterol", "Omeprazole", "Ibuprofen", "Acetaminophen"
        ]
        
        for medication in possible_medications:
            if random.random() < 0.1:
                self.patient_profile["current_medications"].append(medication)
    
    def generate_patient_prompt(self, doctor_question):
        """
        Generate a prompt for the language model to generate a patient response
        
        Args:
            doctor_question: The doctor's question
            
        Returns:
            A prompt for the language model
        """
        # Create the system prompt
        system_prompt = f"""You are roleplaying as a patient with the following condition and symptoms:
Disease: {self.patient_profile['disease']}

Your symptoms include:
{', '.join([s.replace('_', ' ') for s, has in self.patient_profile['symptoms'].items() if has])}

You do NOT have these symptoms:
{', '.join([s.replace('_', ' ') for s, has in self.patient_profile['symptoms'].items() if not has])}

Patient profile:
- Age: {self.patient_profile['age']}
- Gender: {self.patient_profile['gender']}
- Medical history: {', '.join(self.patient_profile['medical_history']) if self.patient_profile['medical_history'] else 'None'}
- Current medications: {', '.join(self.patient_profile['current_medications']) if self.patient_profile['current_medications'] else 'None'}

IMPORTANT RULES:
1. Answer the doctor's questions truthfully based on your symptoms.
2. Only mention symptoms that are in your profile.
3. Do not diagnose yourself - you don't know what your condition is called.
4. Keep your answers brief and direct, as a typical patient would.
5. Express some concern about your condition, but don't be overly dramatic.
6. If you're asked about a symptom not listed above, say you don't have it.
7. Your responses must be 1-3 sentences maximum.
"""
        
        # Create the conversation context
        conv_context = ""
        if self.conversation_history:
            conv_context = "Previous conversation:\n"
            for i, (speaker, message) in enumerate(self.conversation_history):
                conv_context += f"{speaker}: {message}\n"
        
        # Create the final prompt
        final_prompt = f"{system_prompt}\n\n{conv_context}\nDoctor: {doctor_question}\n\nYour response as the patient:"
        
        return final_prompt
    
    def respond(self, doctor_question):
        """
        Generate a patient response to the doctor's question
        
        Args:
            doctor_question: The doctor's question
            
        Returns:
            The patient's response
        """
        # Add the doctor's question to the conversation history
        self.conversation_history.append(("Doctor", doctor_question))
        
        # Generate the prompt
        prompt = self.generate_patient_prompt(doctor_question)
        
        # Generate a response
        from vllm import SamplingParams
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
        )
        
        # Get response from model
        try:
            # First attempt to use fast_generate (for Unsloth optimized models)
            response = self.model.fast_generate(
                [prompt],
                sampling_params=sampling_params,
            )[0].outputs[0].text.strip()
        except AttributeError:
            # Fall back to regular generate method for standard models
            import torch
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode the generated tokens
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Truncate if too long
        if len(response) > 200:
            response = response[:200] + "..."
        
        # Add the response to the conversation history
        self.conversation_history.append(("Patient", response))
        
        return response

def create_llm_patient(model, tokenizer, disease_info=None):
    """
    Create a language model-based patient simulator
    
    Args:
        model: A language model
        tokenizer: The tokenizer for the model
        disease_info: Optional disease information
        
    Returns:
        A LLMPatientSimulator instance
    """
    return LLMPatientSimulator(model, tokenizer, disease_info)

# Wrapper function to make it easy to use the LLM patient in place of the rule-based patient
def llm_patient_response(question, disease_info, conversation_history, model, tokenizer):
    """
    Generate a patient response using a language model
    
    Args:
        question: The doctor's question
        disease_info: Disease information dictionary
        conversation_history: List of previous conversation messages
        model: The language model
        tokenizer: The tokenizer for the model
        
    Returns:
        The patient's response
    """
    # Create a patient simulator
    simulator = LLMPatientSimulator(model, tokenizer, disease_info)
    
    # Convert conversation_history to the format used by the simulator
    if conversation_history:
        for message in conversation_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                simulator.conversation_history.append(("Patient", content))
            elif role == "assistant":
                simulator.conversation_history.append(("Doctor", content))
    
    # Generate a response
    return simulator.respond(question)