# -*- coding: utf-8 -*-
"""
GRPODx: Medical Diagnosis Agent using GRPO (Group Relative Policy Optimization)

This implementation follows the technical specification to create a medical diagnostic 
agent that can conduct multi-turn conversations with patients and make diagnoses.
"""

# Install dependencies
# !pip install unsloth vllm

# Import necessary libraries
import os
# Disable HF transfer to fix download issues
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import random
import json
import re
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from config import MODEL_CONFIG, TRAINING_CONFIG, SYSTEM_PROMPT

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

# Use SYSTEM_PROMPT imported from config.py

def format_conversation(conversation_history, turn_number=None):
    """Format conversation history for model input"""
    system_content = SYSTEM_PROMPT
    
    # Add turn number to system prompt if provided
    if turn_number is not None:
        system_content += f"\n\nCURRENT TURN: {turn_number}/20"
    
    formatted_messages = [
        {"role": "system", "content": system_content}
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

def extract_diagnosis_tag(text):
    """Extract the diagnosis from <diagnosis> tags, with safety checks"""
    # Safety check for None or non-string values
    if text is None:
        return None
    
    # Ensure text is a string
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None
    
    # Search for diagnosis in new tag format
    try:
        diagnosis_tag_pattern = r"<diagnosis>(.*?)</diagnosis>"
        match = re.search(diagnosis_tag_pattern, text, re.DOTALL)
        if match:
            diagnosis_content = match.group(1).strip()
            
            # Now extract the "Final diagnosis: X" from within the tags
            final_pattern = r"Final diagnosis: ([A-Za-z\s\-\.,']+)"
            final_match = re.search(final_pattern, diagnosis_content)
            
            if final_match:
                diagnosis = final_match.group(1).strip()
                # Basic check to filter out clearly non-diagnostic text
                if ("question" in diagnosis.lower() or 
                    "symptom" in diagnosis.lower() or 
                    "next" in diagnosis.lower() or
                    "explore" in diagnosis.lower() or
                    "affecting" in diagnosis.lower() or
                    "towards conditions" in diagnosis.lower()):
                    print(f"Warning: Rejected non-diagnostic text in diagnosis tag: '{diagnosis}'")
                    return None
                return diagnosis
                
            # If no "Final diagnosis:" format found, use the whole content as a fallback
            # But first check if it's likely to be diagnostic text
            if ("question" in diagnosis_content.lower() or 
                "symptom" in diagnosis_content.lower() or 
                "next" in diagnosis_content.lower() or
                "explore" in diagnosis_content.lower() or
                "affecting" in diagnosis_content.lower() or
                "towards conditions" in diagnosis_content.lower()):
                print(f"Warning: Rejected non-diagnostic text in diagnosis tag: '{diagnosis_content}'")
                return None
                
            # Return the whole content if we can't extract using the pattern
            return diagnosis_content.strip()
            
    except Exception as e:
        print(f"Warning: Error extracting diagnosis from tags: {e}")
    
    return None

def is_valid_diagnosis(diagnosis_text):
    """Check if a string looks like a valid diagnosis"""
    if diagnosis_text is None or len(diagnosis_text.strip()) < 3:
        return False
        
    # Immediate rejection patterns - strings that are definitely not diagnoses
    rejection_patterns = [
        r'(considering|consider) (these|this|the) symptoms',
        r'(considering|consider) (which|what) (specific|best)',
        r'(aims|aim|going|need) to (explore|examine|determine|clarify|identify)',
        r'help (differentiate|diagnose|determine|identify)',
        r'(further|additional|more) (questions|assessment|investigation)',
        r'(next|my) (question|assessment)',
        r'(potential|possible|differential) (diagnoses|diagnosis)',
        r'(systems|symptoms|conditions) (further|affecting)',
        r'(respiratory|cardiovascular|neurological|gastrointestinal) (system|symptoms)',
        r'towards conditions affecting',
        r'(towards|suggesting|suggests|indicative of|pointing to)'
    ]
    
    for pattern in rejection_patterns:
        if re.search(pattern, diagnosis_text.lower()):
            print(f"Rejected invalid diagnosis with pattern matching '{pattern}': '{diagnosis_text}'")
            return False
            
    # Must have at least one capitalized word as it should be a disease name
    has_capitalized = any(word[0].isupper() for word in diagnosis_text.split() if word and len(word) > 1)
    if not has_capitalized:
        print(f"Rejected diagnosis without capitalization: '{diagnosis_text}'")
        return False
        
    # Check if it looks like a disease name    
    # Disease name patterns and indicators
    disease_indicators = [
        "itis", "emia", "oma", "pathy", "osis", "disease", "syndrome", 
        "disorder", "infection", "cancer", "fever", "flu", "cold", 
        "hepatitis", "diabetes", "arthritis", "pneumonia", "anemia", 
        "hypertension", "asthma", "migraine", "malaria", "tuberculosis"
    ]
    
    # At least one of these conditions should be true for a valid diagnosis
    conditions = [
        # Contains disease-like indicator word
        any(indicator in diagnosis_text.lower() for indicator in disease_indicators),
        
        # Is short (2-3 words) and has capitalization (likely a proper disease name)
        (len(diagnosis_text.split()) <= 3 and has_capitalized),
        
        # Has a specific disease pattern like "X Disease" or "X Syndrome"
        bool(re.search(r'[A-Z][a-z]+ (Disease|Syndrome|Disorder|Infection)', diagnosis_text))
    ]
    
    if not any(conditions):
        print(f"Rejected diagnosis that doesn't match disease patterns: '{diagnosis_text}'")
        return False
        
    return True

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
    
    # First, try to extract from the new <diagnosis> tags
    diagnosis_from_tag = extract_diagnosis_tag(text)
    if diagnosis_from_tag:
        return diagnosis_from_tag
    
    # If not found in tags, search for diagnosis pattern - primary format we want
    try:
        diagnosis_pattern = r"Final diagnosis: ([A-Za-z\s\-\.,']+)"
        match = re.search(diagnosis_pattern, text)
        if match:
            diagnosis = match.group(1).strip()
            # Filter out vague non-diagnoses that start with "of a" or "a"
            if diagnosis.lower().startswith(("of a", "a ", "of ")):
                # This is likely a description, not a specific disease name
                # Try to find the actual disease name in the text
                disease_name = None
                words = diagnosis.split()
                for i, word in enumerate(words):
                    # Look for capitalized words after the initial "of a" or "a"
                    if i > 1 and word and word[0].isupper():
                        disease_name = " ".join(words[i:])
                        break
                
                if disease_name:
                    return disease_name.strip()
                # Otherwise, return the original but with a note that it's not ideal
                print(f"Warning: Vague diagnosis format detected: '{diagnosis}'")
                return diagnosis
            return diagnosis
    except Exception as e:
        print(f"Warning: Error extracting diagnosis: {e}")
    
    # Alternative patterns in case the format is different
    try:
        # Look for "diagnosis is X" pattern
        alt_pattern1 = r"diagnosis is ([A-Za-z\s\-\.,']+)"
        match = re.search(alt_pattern1, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        # Look for just "diagnosis: X" pattern
        alt_pattern2 = r"diagnosis:?\s*([A-Za-z\s\-\.,']+)"
        match = re.search(alt_pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        # If all else fails, look for capitalized words together that might be a disease name
        # This is a fallback for when the format is completely wrong
        caps_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        matches = re.findall(caps_pattern, text)
        if matches:
            # Return the longest match as it's most likely to be a disease name
            return max(matches, key=len)
    except Exception as e:
        print(f"Warning: Error in alternative diagnosis extraction: {e}")
    
    return None

# Import LLM-based patient simulator
from llm_patient import llm_patient_response, create_llm_patient

# Episode simulation
def run_episode(model, tokenizer, disease_info=None, max_turns=20, use_llm_patient=False, use_verdict=False):
    """
    Run a complete diagnostic episode
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        disease_info: Optional disease info. If None, a new disease will be generated
        max_turns: Maximum number of conversation turns
        use_llm_patient: Whether to use the LLM-based patient simulator
        use_verdict: Whether to use Verdict-based reward scoring (requires OpenAI API key)
        
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
    for turn in range(max_turns):
        # Current turn number (1-indexed)
        current_turn = turn + 1
        
        # Format conversation for model input with turn number
        formatted_conv = format_conversation(conversation, turn_number=current_turn)
        prompt = tokenizer.apply_chat_template(formatted_conv, tokenize=False, add_generation_prompt=True)
        
        # Generate doctor's response
        # Try different generation methods based on what's available
        try:
            # First attempt to use fast_generate (for Unsloth optimized models)
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=1024,  # Doubled from 512 to prevent truncation
            )
            
            # Check if the model has fast_generate attribute
            if hasattr(model, 'fast_generate'):
                response = model.fast_generate(
                    [prompt],
                    sampling_params=sampling_params,
                )[0].outputs[0].text
            else:
                # Skip to regular generate method for models without fast_generate
                raise AttributeError("Model doesn't support fast_generate")
        except (AttributeError, RuntimeError) as e:
            # Fall back to regular generate method for standard models
            print(f"Using standard generate method (model doesn't support fast_generate: {str(e)})")
            import torch
            
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            try:
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            except RuntimeError as err:
                # Handle potential SDPA alignment errors by falling back to safer generation
                print(f"Generation error: {str(err)}")
                print("Trying with safer generation parameters...")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,  # Reduced token count
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                        num_beams=1,  # Disable beam search
                    )
            
            # Decode the generated tokens
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Check for diagnosis first - it takes precedence
        diagnosis = extract_diagnosis_tag(response)
        
        # If no diagnosis found, look for a question
        question_text = None
        if not diagnosis:
            question_text = extract_question(response)
            
            if not question_text:
                # If no valid question format, skip this turn
                continue
                
            # Still check if there's a diagnosis in the question text
            diagnosis = extract_diagnosis(question_text)
        
        # Add doctor's response to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # Track questions for repetition detection (if it's a question)
        if question_text:
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
    
    # Calculate reward using Verdict if enabled, otherwise use the original method
    if use_verdict and final_diagnosis:
        try:
            reward = calculate_verdict_reward(final_diagnosis, disease_info, conversation, num_turns=len(conversation) // 2)
        except Exception as e:
            print(f"Verdict reward calculation failed: {e}")
            print("Falling back to traditional reward calculation.")
            reward = calculate_traditional_reward(final_diagnosis, disease_info, questions_asked)
    else:
        reward = calculate_traditional_reward(final_diagnosis, disease_info, questions_asked)
    
    return conversation, final_diagnosis, reward, disease_info

def calculate_verdict_reward(diagnosis, disease_info, conversation, num_turns=0):
    """
    Calculate reward using Verdict-based scoring system
    
    Args:
        diagnosis: The final diagnosis
        disease_info: The actual disease information
        conversation: The complete conversation history
        num_turns: Number of turns taken
        
    Returns:
        A reward score from 0 to 1
    """
    try:
        # DEBUGGING: Print Python version and path
        import sys
        print(f"Python version: {sys.version}")
        print(f"Python path: {sys.path}")
        
        # Check for local verdict.py or verdict_example.py files that might conflict
        import os
        cwd = os.getcwd()
        local_files = os.listdir(cwd)
        print(f"Files in current directory: {local_files}")
        verdict_files = [f for f in local_files if 'verdict' in f.lower()]
        if verdict_files:
            print(f"WARNING: Found local verdict files that may conflict with imports: {verdict_files}")
            
            # Attempt to rename any verdict.py files to prevent import conflicts
            if 'verdict.py' in local_files:
                backup_name = 'verdict.py.bak'
                counter = 1
                while backup_name in local_files:
                    backup_name = f'verdict.py.bak{counter}'
                    counter += 1
                    
                try:
                    print(f"Renaming verdict.py to {backup_name} to avoid import conflicts")
                    os.rename(os.path.join(cwd, 'verdict.py'), os.path.join(cwd, backup_name))
                    print(f"Successfully renamed verdict.py to {backup_name}")
                except Exception as e:
                    print(f"Failed to rename verdict.py: {e}")
        
        # Modify sys.path to ensure installed packages are used first
        # Completely rebuild sys.path to prioritize site-packages
        print("Modifying sys.path to prioritize installed packages...")
        original_path = list(sys.path)
        
        # Remove current directory from path entirely
        if '' in sys.path:
            sys.path.remove('')
            print("Removed current directory from sys.path")
            
        # Get the site-packages directory
        import site
        site_packages = site.getsitepackages()
        print(f"Site packages directories: {site_packages}")
        
        # Prioritize site-packages at the beginning of sys.path
        for sp in reversed(site_packages):
            if sp not in sys.path:
                sys.path.insert(0, sp)
                print(f"Added {sp} to beginning of sys.path")
                
        print(f"Modified sys.path: {sys.path}")
            
        # Check if verdict is installed and API keys are available
        import importlib.util
        try:
            # Try to find the installed verdict package
            verdict_spec = importlib.util.find_spec("verdict")
            print(f"Verdict spec found: {verdict_spec}")
            if verdict_spec:
                print(f"Verdict origin: {verdict_spec.origin}")
                print(f"Verdict submodule_search_locations: {verdict_spec.submodule_search_locations}")
                
                # Check if the found verdict is a local file rather than installed package
                if os.path.dirname(verdict_spec.origin) == cwd or verdict_spec.origin.endswith('/workspace/GRPOdx/verdict.py'):
                    print("WARNING: Found verdict.py is a local file, not the installed package!")
                    print("Attempting to force import from site-packages...")
                    
                    # Try to import directly from site packages
                    for sp in site_packages:
                        verdict_path = os.path.join(sp, 'verdict')
                        if os.path.exists(verdict_path):
                            print(f"Found verdict in {verdict_path}")
                            if sp not in sys.path:
                                sys.path.insert(0, sp)
                                print(f"Added {sp} to beginning of sys.path")
                            break
        except Exception as e:
            print(f"Error checking for verdict module: {e}")
            verdict_spec = None
        
        # Restore original path at the end, but keep site-packages at front
        print("Restoring original sys.path with site-packages prioritized")
        for path in original_path:
            if path != '' and path not in sys.path:  # Don't add current dir back
                sys.path.append(path)
            
        if verdict_spec is None:
            print("Verdict not installed. To use Verdict-based rewards:")
            print("1. Install with: pip install verdict")
            print("2. Set your OpenAI API key as environment variable: OPENAI_API_KEY")
            return calculate_traditional_reward(diagnosis, disease_info, [])
            
        # Check for API key in environment variables
        if not os.environ.get("OPENAI_API_KEY"):
            print("WARNING: OpenAI API key not found in environment")
            print("Please set your key: export OPENAI_API_KEY='your-api-key'")
            print("Falling back to traditional reward calculation")
            return calculate_traditional_reward(diagnosis, disease_info, [])
        
        # Make sure we don't use local verdict.py by using a direct import from a specific path
        try:
            print("Attempting to import verdict modules using a dynamic approach...")
            
            # First, try to find the verdict package in site-packages
            verdict_in_site = None
            for sp in site_packages:
                verdict_path = os.path.join(sp, 'verdict')
                if os.path.exists(verdict_path) and os.path.isdir(verdict_path):
                    verdict_in_site = verdict_path
                    print(f"Found verdict directory in site-packages: {verdict_path}")
                    break
                    
            if verdict_in_site:
                print(f"Will attempt to import directly from {verdict_in_site}")
                
                # Try the direct import approach
                import importlib
                print("Creating a spec from the found verdict directory")
                verdict_spec = importlib.util.spec_from_file_location("verdict", 
                                os.path.join(verdict_in_site, "__init__.py"))
                verdict = importlib.util.module_from_spec(verdict_spec)
                print("Loading the module using the spec")
                verdict_spec.loader.exec_module(verdict)
                print(f"Successfully imported verdict directly from {verdict_in_site}")
                
                # Now import the rest using normal imports
                print("Now importing specific verdict modules")
                from verdict import Pipeline, Layer
                print("Imported Pipeline and Layer")
                
                from verdict.common.judge import CategoricalJudgeUnit
                print("Imported CategoricalJudgeUnit")
                
                from verdict.scale import DiscreteScale
                print("Imported DiscreteScale")
                
                from verdict.transform import MaxPoolUnit
                print("Imported MaxPoolUnit")
                
                from verdict.schema import Schema
                print("Imported Schema")
                
                print(f"Successfully imported all verdict modules, version: {getattr(verdict, '__version__', 'unknown')}")
            else:
                # Fallback to regular imports if we couldn't find verdict in site-packages
                print("Could not find verdict in site-packages. Attempting standard imports...")
                
                # Try to use absolute imports
                print("First importing the main verdict module")
                import verdict
                print(f"Successfully imported verdict main module")
                
                print("Now importing specific verdict modules")
                from verdict import Pipeline, Layer
                print("Imported Pipeline and Layer")
                
                from verdict.common.judge import CategoricalJudgeUnit
                print("Imported CategoricalJudgeUnit")
                
                from verdict.scale import DiscreteScale
                print("Imported DiscreteScale")
                
                from verdict.transform import MaxPoolUnit
                print("Imported MaxPoolUnit")
                
                from verdict.schema import Schema
                print("Imported Schema")
                
                print(f"Successfully imported all verdict modules, version: {getattr(verdict, '__version__', 'unknown')}")
        except ImportError as e:
            print(f"Error importing verdict modules: {e}")
            # Print traceback for more debugging info
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return calculate_traditional_reward(diagnosis, disease_info, [])
        
        # Disable rate limiting for local testing
        try:
            print("Trying to import and disable rate limit...")
            from verdict.util import ratelimit
            print("Successfully imported ratelimit")
            ratelimit.disable()
            print("Successfully disabled rate limiting")
        except ImportError as e:
            print(f"Failed to import ratelimit module: {e}")
            import traceback
            print("Full traceback for ratelimit import error:")
            traceback.print_exc()
            # Continue without disabling rate limiting
        
        # Use GPT-4o for better diagnostic evaluation and direct scoring
        model_to_use = "gpt-4o"
        print("Using GPT-4o for diagnostic scoring...")
        
        # Format the disease information as a document
        disease_doc = f"Disease: {disease_info['disease_name']}\n"
        disease_doc += "Symptoms:\n"
        for symptom, has_symptom in disease_info['symptoms'].items():
            if has_symptom:
                disease_doc += f"- {symptom.replace('_', ' ')}: Present\n"
            else:
                disease_doc += f"- {symptom.replace('_', ' ')}: Not present\n"
                
        # Format the conversation and diagnosis as the claim
        conv_summary = "Patient symptoms based on conversation:\n"
        for msg in conversation:
            if msg["role"] == "user":
                patient_msg = msg["content"].strip()
                if not patient_msg.startswith("I already answered") and "ask me something else" not in patient_msg:
                    conv_summary += f"- {patient_msg}\n"
                    
        claim = f"{conv_summary}\nDiagnosis: {diagnosis}"
        
        # Check if IntervalScale is available
        # In newer versions of verdict, it might be using a different name or approach
        try:
            # Try to import IntervalScale first
            print("Checking for IntervalScale in verdict.scale...")
            from verdict.scale import IntervalScale
            print("Successfully imported IntervalScale")
            
            # If available, import IntervalJudgeUnit too
            from verdict.common.judge import IntervalJudgeUnit
            print("Successfully imported IntervalJudgeUnit")
            
            # Now use these classes to create the judge
            interval_scale_available = True
        except ImportError as e:
            print(f"IntervalScale not available in this verdict version: {e}")
            print("Will use CategoricalJudgeUnit with custom scoring instead")
            interval_scale_available = False
        
        if interval_scale_available:
            # Use IntervalJudgeUnit if available
            diagnostic_judge = IntervalJudgeUnit(
                name='DiagnosticScorer', 
                scale=IntervalScale([0.0, 1.0]),
                explanation=True
            ).prompt("""
                Evaluate the diagnostic quality of the provided diagnosis against the disease information.
                Score the diagnosis on a scale from 0.0 to 1.0, where:
                
                - 1.0: Perfect diagnosis that exactly matches the correct disease
                - 0.8-0.9: Very strong diagnosis that identifies the correct disease with minor imprecision
                - 0.6-0.7: Good diagnosis that identifies a closely related condition
                - 0.4-0.5: Partial diagnosis that identifies some aspects of the condition
                - 0.2-0.3: Poor diagnosis with minimal connection to the correct disease
                - 0.0-0.1: Completely incorrect diagnosis
                
                Disease Information: {source.doc}
                Diagnostic Process and Conclusion: {source.claim}
                
                Consider:
                1. Accuracy: How well does the diagnosis match the actual disease?
                2. Completeness: Does the diagnosis capture all key aspects of the disease?
                3. Specificity: Is the diagnosis appropriately specific (not too vague)?
                4. Clinical Relevance: Would the diagnosis lead to appropriate management?
                
                Score:
            """).via(policy_or_name=model_to_use, retries=1, temperature=0.2)
        else:
            # Fallback to CategoricalJudgeUnit with custom prompt for numerical scoring
            from verdict.scale import DiscreteScale
            
            # Create a discrete scale with a numeric scoring system
            diagnostic_judge = CategoricalJudgeUnit(
                name='DiagnosticScorer',
                categories=DiscreteScale(['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.0']),
                explanation=True
            ).prompt("""
                Evaluate the diagnostic quality of the provided diagnosis against the disease information.
                Score the diagnosis on a scale from 0.0 to 1.0, where:
                
                - 1.0: Perfect diagnosis that exactly matches the correct disease
                - 0.8-0.9: Very strong diagnosis that identifies the correct disease with minor imprecision
                - 0.6-0.7: Good diagnosis that identifies a closely related condition
                - 0.4-0.5: Partial diagnosis that identifies some aspects of the condition
                - 0.2-0.3: Poor diagnosis with minimal connection to the correct disease
                - 0.0-0.1: Completely incorrect diagnosis
                
                Disease Information: {source.doc}
                Diagnostic Process and Conclusion: {source.claim}
                
                Consider:
                1. Accuracy: How well does the diagnosis match the actual disease?
                2. Completeness: Does the diagnosis capture all key aspects of the disease?
                3. Specificity: Is the diagnosis appropriately specific (not too vague)?
                4. Clinical Relevance: Would the diagnosis lead to appropriate management?
                
                You MUST select one of these exact values as your response: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, or 0.0.
                
                Score:
            """).via(policy_or_name=model_to_use, retries=1, temperature=0.2)
        
        # Create a pipeline with just the single scoring judge
        pipeline = Pipeline() >> diagnostic_judge
        
        # Create the test sample
        test_sample = Schema.of(doc=disease_doc, claim=claim)
        
        # Run the verdict pipeline
        response, _ = pipeline.run(test_sample, max_workers=1)
        
        # Extract the score from the response (either score or choice depending on judge type)
        print(f"Response keys from verdict: {list(response.keys())}")
        
        try:
            if interval_scale_available:
                # When using IntervalJudgeUnit, look for '_score' or '_value' keys
                score_key = list(filter(lambda k: '_score' in k or '_value' in k, response.keys()))[0]
                print(f"Found score key: {score_key}")
                
                # Get the raw score from the judge
                judge_score = float(response[score_key])
                print(f"Raw score from IntervalJudgeUnit: {judge_score}")
            else:
                # When using CategoricalJudgeUnit, look for '_choice' key
                choice_key = list(filter(lambda k: '_choice' in k, response.keys()))[0]
                print(f"Found choice key: {choice_key}")
                
                # Convert the string choice to a float
                judge_score = float(response[choice_key])
                print(f"Score from CategoricalJudgeUnit: {judge_score}")
            
            # Ensure it's within valid range
            base_reward = max(0.0, min(1.0, judge_score))
            
            # Round to 2 decimal places for clarity
            base_reward = round(base_reward, 2)
            print(f"Final base reward after validation: {base_reward}")
            
            # Extract the explanation if available
            explanation_keys = list(filter(lambda k: '_explanation' in k, response.keys()))
            if explanation_keys:
                explanation_key = explanation_keys[0]
                if explanation_key in response:
                    print(f"Verdict explanation: {response[explanation_key]}")
        except (ValueError, TypeError, IndexError) as e:
            # Fall back to a default method if score parsing fails
            print(f"Failed to parse score from Verdict: {e}")
            print("Response contents:", response)
            print("Using fallback scoring")
            
            # Try to find any numerical values in the response that might represent a score
            for key, value in response.items():
                if isinstance(value, str) and any(x in value for x in ['0.', '1.', '0,', '1,']):
                    print(f"Found potential score in {key}: {value}")
                    try:
                        # Try to extract a float from the string
                        import re
                        match = re.search(r'(0\.\d+|1\.0|1)', value)
                        if match:
                            base_reward = float(match.group(0))
                            print(f"Extracted score: {base_reward}")
                            break
                    except:
                        pass
            
            # If still no score found, check for yes/no or try word overlap
            if 'base_reward' not in locals():
                # Find any 'yes'/'no' assessment that might exist
                yes_no_key = list(filter(lambda k: '_choice' in k, response.keys()))
                if yes_no_key and response[yes_no_key[0]] == 'yes':
                    base_reward = 0.85  # Default high score for 'yes'
                else:
                    # Check for partial match based on word overlap
                    disease_words = set(disease_info["disease_name"].lower().split())
                    diagnosis_words = set(diagnosis.lower().split())
                    common_words = disease_words.intersection(diagnosis_words)
                    
                    if len(common_words) > 0 and len(common_words) >= len(disease_words) / 3:
                        base_reward = 0.4  # Partial credit for related diagnosis
                    else:
                        base_reward = 0.1  # Minimal credit
        
        # Add speed bonus for both correct and partially correct diagnoses
        speed_bonus = 0.0
        if base_reward > 0:
            if num_turns <= 5:
                speed_bonus = 0.25  # Maximum speed bonus
            else:
                max_efficient_turns = 15
                if num_turns <= max_efficient_turns:
                    speed_bonus = 0.25 * (1 - (num_turns - 5) / (max_efficient_turns - 5))
        
        final_reward = base_reward + speed_bonus
        
        # Always provide some minimal reward for making any diagnosis
        if final_reward == 0.0 and diagnosis:
            final_reward = 0.1
        
        return final_reward
        
    except Exception as e:
        # Fall back to traditional reward calculation if verdict fails
        print(f"Verdict reward calculation failed: {e}")
        print("Detailed exception information:")
        import traceback
        traceback.print_exc()
        print("Falling back to traditional reward calculation.")
        return calculate_traditional_reward(diagnosis, disease_info, [])

def calculate_traditional_reward(final_diagnosis, disease_info, questions_asked):
    """
    Calculate reward using the traditional method (original implementation)
    
    Args:
        final_diagnosis: The final diagnosis
        disease_info: The actual disease information
        questions_asked: List of questions asked during the episode
        
    Returns:
        A reward score from 0 to 1
    """
    reward = 0.0
    
    # Get the number of turns taken (length of conversation divided by 2)
    num_turns = len(questions_asked) + 1 if final_diagnosis else len(questions_asked)
    
    # Exact match reward
    if final_diagnosis:
        # Base reward for providing any diagnosis at all (even if it's wrong)
        reward = 0.2
        
        # Add an extra 0.1 if the diagnosis has an actual disease name (not just a description)
        # This encourages specific diagnoses over vague ones
        if any(word[0].isupper() for word in final_diagnosis.split()):
            reward += 0.1
            
        if final_diagnosis.lower() == disease_info["disease_name"].lower():
            # Exact match gets variable reward based on speed
            # Start with base reward of 1.0
            base_reward = 1.0
            
            # Calculate speed bonus - earlier diagnoses get higher rewards
            # Maximum bonus for diagnoses in 5 turns or fewer
            if num_turns <= 5:
                speed_bonus = 0.5  # Maximum speed bonus
            else:
                # Linearly decrease bonus as turns increase, up to turn 15
                max_efficient_turns = 15
                if num_turns <= max_efficient_turns:
                    speed_bonus = 0.5 * (1 - (num_turns - 5) / (max_efficient_turns - 5))
                else:
                    speed_bonus = 0.0  # No speed bonus after max_efficient_turns
            
            # Apply total reward
            reward = base_reward + speed_bonus
            
        else:
            # Partial match reward - if disease name contains parts of real diagnosis
            disease_words = set(disease_info["disease_name"].lower().split())
            diagnosis_words = set(final_diagnosis.lower().split())
            common_words = disease_words.intersection(diagnosis_words)
            
            if len(common_words) > 0 and len(common_words) >= len(disease_words) / 3:
                reward = 0.5  # Increased partial credit for related diagnosis
                
                # Still give a small speed bonus for partial matches
                if num_turns <= 5:
                    speed_bonus = 0.2  # Smaller speed bonus for partial matches
                else:
                    max_efficient_turns = 15
                    if num_turns <= max_efficient_turns:
                        speed_bonus = 0.2 * (1 - (num_turns - 5) / (max_efficient_turns - 5))
                    else:
                        speed_bonus = 0.0
                
                reward += speed_bonus
    
    # Penalize for question repetition
    unique_questions = set([q.lower() for q in questions_asked])
    if len(questions_asked) > len(unique_questions):
        repetition_penalty = 0.1 * (len(questions_asked) - len(unique_questions))
        reward = max(0, reward - repetition_penalty)
    
    return reward

# Prepare dataset for GRPO
def generate_training_batch(model, tokenizer, batch_size=4, completions_per_scenario=6, verbose=True, use_llm_patient=False, use_verdict=False):
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
                use_llm_patient=use_llm_patient,
                use_verdict=use_verdict
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
    def __init__(self, model, tokenizer, print_frequency=10, use_llm_patient=False, use_verdict=False):
        self.model = model
        self.tokenizer = tokenizer
        self.print_frequency = print_frequency
        self.step = 0
        self.example_scenarios = []
        self.last_rewards = []
        self.use_llm_patient = use_llm_patient
        self.use_verdict = use_verdict
    
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
                        self.model, self.tokenizer, disease, max_turns=16,
                        use_llm_patient=self.use_llm_patient,
                        use_verdict=self.use_verdict
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
                        print(f"Turns taken: {len(conversation) // 2}")
                        print(f"Reward: {reward:.2f}")
                    else:
                        print("\nNo final diagnosis provided")
                        print(f"Turns taken: {len(conversation) // 2}")
                        print(f"Reward: {reward:.2f}")
                except Exception as e:
                    print(f"Error generating test conversation: {e}")
                
                print(f"{'='*90}\n")
        
        return control

# Main training function
def train_grpodx(num_steps=500, batch_size=4, completions_per_scenario=6, verbose=True, use_llm_patient=False, use_verdict=False):
    """
    Main training function for GRPODx
    
    Args:
        num_steps: Number of training steps
        batch_size: Number of different diseases per batch
        completions_per_scenario: Number of completions per disease for GRPO
        verbose: Whether to print detailed logs during training
        use_llm_patient: Whether to use the LLM-based patient simulator
        use_verdict: Whether to use Verdict-based reward scoring (requires OpenAI API key)
    """
    # Load model parameters from config
    max_seq_length = MODEL_CONFIG["max_seq_length"]
    lora_rank = MODEL_CONFIG.get("lora_rank", 8)
    
    # Create a global disease cache that can be shared with reward functions
    disease_cache = []
    
    # Load model with fallbacks from config
    model_options = MODEL_CONFIG["model_options"]
    
    # Try models in order until one works
    for model_name in model_options:
        try:
            print(f"Attempting to load model: {model_name}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=MODEL_CONFIG.get("load_in_4bit", True),
                fast_inference=MODEL_CONFIG.get("fast_inference", True),
                max_lora_rank=lora_rank,
                gpu_memory_utilization=MODEL_CONFIG.get("gpu_memory_utilization", 0.9),
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
    max_prompt_length = TRAINING_CONFIG.get("max_prompt_length", 1024)
    
    training_args = GRPOConfig(
        learning_rate=TRAINING_CONFIG.get("learning_rate", 2e-6),
        adam_beta1=TRAINING_CONFIG.get("adam_beta1", 0.9),
        adam_beta2=TRAINING_CONFIG.get("adam_beta2", 0.99),
        weight_decay=TRAINING_CONFIG.get("weight_decay", 0.1),
        warmup_ratio=TRAINING_CONFIG.get("warmup_ratio", 0.1),
        lr_scheduler_type=TRAINING_CONFIG.get("lr_scheduler_type", "cosine"),
        optim=TRAINING_CONFIG.get("optim", "paged_adamw_8bit"),
        logging_steps=TRAINING_CONFIG.get("logging_steps", 1),
        per_device_train_batch_size=TRAINING_CONFIG.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=TRAINING_CONFIG.get("gradient_accumulation_steps", 2),
        num_generations=completions_per_scenario,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        max_steps=num_steps,
        save_steps=num_steps // 5,
        max_grad_norm=TRAINING_CONFIG.get("max_grad_norm", 0.1),
        report_to="none",
        output_dir=TRAINING_CONFIG.get("output_dir", "outputs"),
    )
    
    # Generate initial training dataset
    initial_dataset = generate_training_batch(
        model, 
        tokenizer, 
        batch_size, 
        completions_per_scenario, 
        verbose=verbose,
        use_llm_patient=use_llm_patient,
        use_verdict=use_verdict
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
                # Even a vague diagnosis should get something, to encourage diagnosis over no diagnosis
                reward = 0.2
                
                # Add an extra 0.1 if the diagnosis has an actual disease name (not just a description)
                # This encourages specific diagnoses over vague ones
                if any(word[0].isupper() for word in diagnosis.split()):
                    reward += 0.1
                
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
        callback = TrainingCallback(model, tokenizer, print_frequency=20, 
                                   use_llm_patient=use_llm_patient, 
                                   use_verdict=use_verdict)
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
        print(f"[Reward System: Base reward 1.0 for correct diagnosis + up to 0.5 bonus for speed]")
        print(f"[Speed Bonus: Maximum +0.5 for diagnosis in 5 turns or fewer, decreasing to 0 after 15 turns]")
        
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
    max_turns = 30  # Doubled from 15 to prevent infinite loops
    
    while turn_count < max_turns:
        turn_count += 1
        
        # Format conversation for model input with turn number
        formatted_conv = format_conversation(conversation, turn_number=turn_count)
        prompt = tokenizer.apply_chat_template(formatted_conv, tokenize=False, add_generation_prompt=True)
        
        # Generate doctor's response
        try:
            # First attempt to use fast_generate (for Unsloth optimized models)
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=1024,  # Doubled from 512 to prevent truncation
            )
            
            lora_params = None
            try:
                lora_params = model.load_lora("grpodx_model")
            except:
                print("Warning: Unable to load LoRA weights with fast_generate method")
            
            response = model.fast_generate(
                [prompt],
                sampling_params=sampling_params,
                lora_request=lora_params,
            )[0].outputs[0].text
            
        except AttributeError:
            # Fall back to regular generate method for standard models
            print("Using standard generate method (model doesn't support fast_generate)")
            import torch
            
            # Try to load LoRA weights if not already loaded
            try:
                # Check if model has peft_config attribute (indicating LoRA is loaded)
                if not hasattr(model, 'peft_config'):
                    print("Loading LoRA weights with PEFT")
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, "grpodx_model")
            except Exception as e:
                print(f"Warning: Unable to load LoRA weights: {e}")
            
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode the generated tokens
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract parts
        reasoning_text = extract_reasoning(response)
        
        # Check for diagnosis first (in tags)
        diagnosis = extract_diagnosis_tag(response)
        
        # If it's a diagnosis
        if diagnosis:
            # Display diagnosis with turn number
            print(f"Doctor (Turn {turn_count}/{max_turns}): Final diagnosis: {diagnosis}")
            
            # Always show reasoning for diagnoses
            if reasoning_text:
                print(f"\n[Doctor's reasoning: {reasoning_text}]\n")
        
        # If not a diagnosis, it's a question
        else:
            question_text = extract_question(response)
            
            if not question_text:
                print("Doctor: I need more information to make a diagnosis.")
                question_text = "Can you tell me more about your symptoms?"
            else:
                # Still check if there's a diagnosis in the question (old format)
                diagnosis = extract_diagnosis(question_text)
                
                if not diagnosis:
                    # Display doctor's question with turn number
                    print(f"Doctor (Turn {turn_count}/{max_turns}): {question_text}")
                    
                    # Optionally show reasoning
                    if reasoning_text and random.random() < 0.3:  # Show reasoning occasionally
                        print(f"\n[Doctor's reasoning: {reasoning_text}]\n")
                else:
                    # It was a diagnosis in the question format
                    print(f"Doctor (Turn {turn_count}/{max_turns}): Final diagnosis: {diagnosis}")
                    
                    # Always show reasoning for diagnoses
                    if reasoning_text:
                        print(f"\n[Doctor's reasoning: {reasoning_text}]\n")
        
        # Add to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # Check if we found a diagnosis and need to end the conversation
        if diagnosis:
            # In simulation mode, verify the diagnosis and calculate reward
            if simulation_mode:
                # Calculate a simulated reward based on speed
                speed_bonus = 0.0
                if turn_count <= 5:
                    speed_bonus = 0.5  # Maximum speed bonus
                elif turn_count <= 15:
                    speed_bonus = 0.5 * (1 - (turn_count - 5) / 10)  # Linear decrease
                
                if diagnosis.lower() == simulated_disease["disease_name"].lower():
                    reward = 1.0 + speed_bonus
                    print(f"\n[Correct diagnosis! ✓]")
                    print(f"[Turns taken: {turn_count}, Reward: {reward:.2f}]")
                    print(f"[Speed bonus: +{speed_bonus:.2f} for diagnosing in {turn_count} turns]")
                else:
                    reward = 0.2  # Base reward for any diagnosis
                    print(f"\n[Incorrect diagnosis. The actual disease was: {simulated_disease['disease_name']}]")
                    print(f"[Turns taken: {turn_count}, Reward: {reward:.2f}]")
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
    import argparse
    parser = argparse.ArgumentParser(description='Train and run GRPODx')
    parser.add_argument('--use-verdict', action='store_true', help='Use Verdict for reward scoring')
    parser.add_argument('--steps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode after training')
    args = parser.parse_args()
    
    # Default to using Verdict (requires OPENAI_API_KEY environment variable)
    args.use_verdict = True if os.environ.get("OPENAI_API_KEY") else False
    
    # Display Verdict information if enabled
    if args.use_verdict:
        print("\n=== Using Verdict with GPT-4o for reward scoring ===")
        print("Using OPENAI_API_KEY from environment variables")
        print("Using GPT-4o for accurate diagnostic scoring")
        print("=================================================\n")
    
    # Train the model
    model, tokenizer = train_grpodx(num_steps=args.steps, use_verdict=args.use_verdict)
    
    # Run interactive session if requested
    if args.interactive:
        interactive_diagnosis(model, tokenizer)