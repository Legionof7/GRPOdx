"""
Test scenarios for evaluating the GRPODx model.

This file contains test cases and evaluation functions for the medical diagnosis agent.
"""

import random
from GRPODx_implementation import (
    format_conversation, extract_diagnosis, extract_question, DISEASE_EXAMPLES,
    calculate_verdict_reward, calculate_traditional_reward
)
from disease_generator import (
    generate_random_disease, generate_disease_batch, generate_related_diseases
)
from vllm import SamplingParams

# Generate test disease bank for evaluation
TEST_DISEASE_BANK = generate_disease_batch(10)

# Add some specific test cases to ensure variety
TEST_DISEASE_BANK.append({
    "disease_name": "Migraine",
    "symptoms": {
        "headache": True,
        "nausea": True,
        "vomiting": False,
        "light_sensitivity": True,
        "sound_sensitivity": True,
        "aura": False,
        "fever": False,
        "cough": False,
        "sore_throat": False,
        "body_aches": False,
        "fatigue": True,
        "runny_nose": False,
    }
})

# Generate a set of related diseases to test differential diagnosis ability
base_disease = generate_random_disease()
related_diseases = generate_related_diseases(base_disease, 3)
TEST_DISEASE_BANK.extend(related_diseases)

# Keep a reference to the base disease for reporting
BASE_DISEASE_FOR_RELATED_TESTS = base_disease

def simulate_patient_response(question, disease_info, conversation_history=None):
    """
    Simulate a patient response based on the disease info and conversation history
    
    This is a wrapper for the main patient_response function in GRPODx_implementation.py
    """
    # Import the main patient response function from GRPODx_implementation
    from GRPODx_implementation import patient_response
    
    # Use the enhanced patient response function
    return patient_response(question, disease_info, conversation_history)

def run_test_episode(model, tokenizer, disease_info=None, max_turns=20, verbose=True, use_verdict=False):
    """
    Run a diagnostic test episode with the trained model
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        disease_info: Optional disease info dictionary. If None, a new disease will be generated
        max_turns: Maximum number of conversation turns
        verbose: Whether to print the conversation
        use_verdict: Whether to use Verdict-based reward scoring
        
    Returns:
        Dictionary with evaluation results
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
    
    if verbose:
        print(f"Patient: {initial_message}")
    conversation.append({"role": "user", "content": initial_message})
    
    final_diagnosis = None
    questions_asked = []
    
    # Diagnostic conversation loop
    for turn in range(max_turns):
        # Format conversation for model input with turn number
        current_turn = turn + 1
        formatted_conv = format_conversation(conversation, turn_number=current_turn) 
        prompt = tokenizer.apply_chat_template(formatted_conv, tokenize=False, add_generation_prompt=True)
        
        # Generate doctor's response
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,  # Increased to allow longer responses
        )
        
        response = model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
        )[0].outputs[0].text
        
        # Extract parts
        question_text = extract_question(response)
        
        if not question_text:
            # If no valid question format, skip this turn
            question_text = "Can you tell me more about your symptoms?"
            if verbose:
                print("Doctor: I need more information to make a diagnosis.")
        
        if verbose:
            print(f"Doctor (Turn {turn+1}): {question_text}")
        
        # Check if this is a final diagnosis
        diagnosis = extract_diagnosis(question_text)
        
        # Add doctor's response to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # Track questions to detect repetition
        questions_asked.append(question_text)
        
        if diagnosis:
            final_diagnosis = diagnosis
            break
        
        # Generate patient's response
        patient_reply = simulate_patient_response(question_text, disease_info, conversation)
        if verbose:
            print(f"Patient: {patient_reply}")
        conversation.append({"role": "user", "content": patient_reply})
    
    # Evaluate performance
    accuracy = 0
    repetition_penalty = 0
    efficiency_score = 0
    symptom_coverage = 0
    
    # Use verdict-based or traditional scoring for accuracy
    if use_verdict and final_diagnosis:
        # Display using verdict if enabled
        if verbose:
            print("\nUsing Verdict with GPT-4o for scoring...")
        
        # Get verdict score from GPT-4o
        verdict_score = calculate_verdict_reward(final_diagnosis, disease_info, conversation, num_turns=turn+1)
        
        # Map verdict score range (typically 0.0-1.0) to accuracy
        # A higher verdict score should represent a better diagnosis
        accuracy = verdict_score * 0.8  # Scale to be comparable with traditional accuracy
        
        if verbose:
            print(f"Verdict score: {verdict_score:.2f}")
    else:
        # Traditional accuracy scoring
        if final_diagnosis:
            if final_diagnosis.lower() == disease_info["disease_name"].lower():
                accuracy = 1.0
            else:
                # Partial match - word overlap
                disease_words = set(disease_info["disease_name"].lower().split())
                diagnosis_words = set(final_diagnosis.lower().split())
                common_words = disease_words.intersection(diagnosis_words)
                
                if len(common_words) > 0 and len(common_words) >= len(disease_words) / 3:
                    accuracy = 0.3  # Partial credit for related diagnosis
    
    # Repetition penalty
    unique_questions = set([q.lower() for q in questions_asked])
    if len(questions_asked) > 0:
        repetition_penalty = (len(questions_asked) - len(unique_questions)) / len(questions_asked)
    
    # Efficiency score (based on number of turns needed)
    if accuracy > 0:
        efficiency_score = 1.0 - (turn / max_turns)
    
    # Symptom coverage - how many of the positive symptoms were asked about
    positive_symptoms = [s.lower() for s, has in disease_info["symptoms"].items() if has]
    asked_about_symptoms = set()
    
    for question in questions_asked:
        for symptom in positive_symptoms:
            if symptom.replace("_", " ") in question.lower():
                asked_about_symptoms.add(symptom)
    
    if positive_symptoms:
        symptom_coverage = len(asked_about_symptoms) / len(positive_symptoms)
    
    # Overall performance score
    performance = (accuracy * 0.4) + (efficiency_score * 0.2) + (symptom_coverage * 0.3) - (repetition_penalty * 0.1)
    
    # Create detailed result
    result = {
        "disease": disease_info["disease_name"],
        "final_diagnosis": final_diagnosis,
        "accuracy": accuracy,
        "efficiency": efficiency_score,
        "repetition_penalty": repetition_penalty,
        "symptom_coverage": symptom_coverage,
        "performance_score": performance,
        "turns_taken": turn + 1,
        "max_turns": max_turns,
        "symptoms_asked": list(asked_about_symptoms),
        "positive_symptoms": positive_symptoms,
    }
    
    return result

def evaluate_model(model, tokenizer, test_cases=None, num_cases=5, use_verdict=False):
    """
    Evaluate the model on multiple test cases
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        test_cases: Optional list of disease cases to test. If None, diseases will be generated
        num_cases: Number of test cases if generating new ones
        use_verdict: Whether to use Verdict-based reward scoring with GPT-4o
        
    Returns:
        Dictionary with evaluation results
    """
    if test_cases is None:
        # Use the test disease bank or generate new cases
        if len(TEST_DISEASE_BANK) >= num_cases:
            test_cases = random.sample(TEST_DISEASE_BANK, num_cases)
        else:
            test_cases = generate_disease_batch(num_cases)
            
            # Add the base disease and one related disease to test differential diagnosis
            if random.random() > 0.5 and len(TEST_DISEASE_BANK) > 0:
                test_cases = test_cases[:-2]  # Remove two to make space
                test_cases.append(BASE_DISEASE_FOR_RELATED_TESTS)
                test_cases.append(random.choice(related_diseases))
    
    results = []
    total_accuracy = 0
    total_efficiency = 0
    total_performance = 0
    total_symptom_coverage = 0
    
    print("===== GRPODx Model Evaluation =====")
    print(f"Running {len(test_cases)} test cases...")
    print(f"Reward system: {'Verdict with GPT-4o' if use_verdict else 'Traditional scoring'}\n")
    
    for i, disease_info in enumerate(test_cases):
        print(f"\n\n===== Test Case {i+1}: {disease_info['disease_name']} =====")
        print(f"Positive Symptoms: {', '.join([s.replace('_', ' ') for s, has in disease_info['symptoms'].items() if has])}")
        
        result = run_test_episode(model, tokenizer, disease_info, use_verdict=use_verdict)
        results.append(result)
        
        total_accuracy += result["accuracy"]
        total_efficiency += result["efficiency"]
        total_performance += result["performance_score"]
        total_symptom_coverage += result["symptom_coverage"]
        
        print(f"\nDiagnosis: {result['final_diagnosis'] or 'None'}")
        print(f"Correct: {'Yes' if result['accuracy'] > 0 else 'No'}")
        print(f"Turns: {result['turns_taken']} / {result['max_turns']}")
        print(f"Symptom Coverage: {result['symptom_coverage']:.2f} ({len(result['symptoms_asked'])}/{len(result['positive_symptoms'])})")
        print(f"Performance Score: {result['performance_score']:.2f}")
    
    # Calculate average scores
    avg_accuracy = total_accuracy / len(test_cases)
    avg_efficiency = total_efficiency / len(test_cases)
    avg_performance = total_performance / len(test_cases)
    avg_symptom_coverage = total_symptom_coverage / len(test_cases)
    
    print("\n===== Overall Evaluation Results =====")
    print(f"Average Accuracy: {avg_accuracy:.2f}")
    print(f"Average Efficiency: {avg_efficiency:.2f}")
    print(f"Average Symptom Coverage: {avg_symptom_coverage:.2f}")
    print(f"Average Performance Score: {avg_performance:.2f}")
    
    return {
        "results": results,
        "avg_accuracy": avg_accuracy,
        "avg_efficiency": avg_efficiency,
        "avg_symptom_coverage": avg_symptom_coverage,
        "avg_performance": avg_performance
    }

# Example usage:
# from GRPODx_implementation import train_grpodx
# model, tokenizer = train_grpodx()
# evaluation = evaluate_model(model, tokenizer)