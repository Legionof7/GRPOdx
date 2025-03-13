import openai
import re
from typing import Dict, Optional

def evaluate_diagnosis(response: str, disease_info: Dict, verbose: bool = False) -> float:
    """
    Evaluate a diagnosis using GPT-4o-mini.
    
    Args:
        response: The doctor's full response
        disease_info: Dictionary containing disease information
        verbose: Whether to print evaluation details
    
    Returns:
        Evaluation score between 0.0 and 1.0
    """
    # Get symptoms present and absent
    symptoms_present = [s for s, v in disease_info['symptoms'].items() if v]
    symptoms_absent = [s for s, v in disease_info['symptoms'].items() if not v]
    
    # Extract diagnosis if possible
    diagnosis = None
    diagnosis_match = re.search(r"final diagnosis:\s*([^.\n]+)", response.lower())
    if diagnosis_match:
        diagnosis = diagnosis_match.group(1).strip()
    else:
        # Try to extract from the <answer> tag
        answer_match = re.search(r"<answer>\s*([^<]+)", response)
        if answer_match:
            diagnosis = answer_match.group(1).strip()
    
    # Check formatting
    has_reasoning = "<reasoning>" in response and "</reasoning>" in response
    has_answer = "<answer>" in response and "</answer>" in response
    has_diagnosis_format = "final diagnosis:" in response.lower()
    
    # Construct prompt for GPT-4o-mini
    system_message = """You are a medical evaluation system. Your task is to evaluate a doctor's diagnosis 
given a set of symptoms and the correct diagnosis. Provide a score from 0.0 to 1.0, where:
- 1.0: Perfect diagnosis that matches the correct disease exactly
- 0.75-0.99: Very good diagnosis that identifies the correct disease with minor inaccuracies
- 0.5-0.74: Partial match - identifies some aspects of the disease or a related condition
- 0.25-0.49: Poor match but shows some understanding of symptoms
- 0.1-0.24: Poor diagnosis with minimal relevant information
- 0.0: Completely incorrect diagnosis or no diagnosis provided

Also consider the format of the response in your evaluation:
- +0.1-0.2: If the response uses proper <reasoning> tags to show diagnostic thinking
- +0.1-0.2: If the response uses proper <answer> tags for the final diagnosis
- +0.05-0.1: If the response includes "Final diagnosis:" format

The total score should not exceed 1.0.

Respond ONLY with a single decimal number between 0.0 and 1.0. Do not include any explanation."""

    user_message = f"""Correct Diagnosis: {disease_info['disease_name']}
Symptoms Present: {', '.join(symptoms_present)}
Symptoms Absent: {', '.join(symptoms_absent)}

Doctor's Response:
{response}

Evaluate and provide only a single number from 0.0 to 1.0:"""
    
    if verbose:
        print("Evaluating with GPT-4o-mini...")
        
    try:
        # Call GPT-4o-mini for evaluation
        api_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=10,
        )
        
        # Extract the score from the response
        score_text = api_response.choices[0].message.content.strip()
        
        # Try to convert to float
        try:
            score = float(score_text)
            # Ensure the score is within bounds
            score = max(0.0, min(1.0, score))
        except ValueError:
            # If not a valid float, use rule-based scoring
            if verbose:
                print(f"Could not parse GPT response as float: {score_text}")
            score = rule_based_score(response, disease_info, has_reasoning, has_answer, has_diagnosis_format, diagnosis)
    
    except Exception as e:
        # Fall back to rule-based scoring
        if verbose:
            print(f"GPT-4o-mini evaluation failed: {e}")
        score = rule_based_score(response, disease_info, has_reasoning, has_answer, has_diagnosis_format, diagnosis)
    
    if verbose:
        print(f"Evaluation score: {score:.2f}")
    
    return score

def rule_based_score(response: str, disease_info: Dict, has_reasoning: bool, has_answer: bool, 
                    has_diagnosis_format: bool, diagnosis: Optional[str]) -> float:
    """Calculate a rule-based score if GPT evaluation fails."""
    # Start with base score
    score = 0.1
    
    # Format points
    if has_reasoning:
        score += 0.15
    if has_answer:
        score += 0.1
    if has_diagnosis_format:
        score += 0.05
    
    # Content points based on diagnosis
    if diagnosis:
        if disease_info["disease_name"].lower() in diagnosis.lower():
            score += 0.6
        else:
            # Check for partial matches
            for term in disease_info["disease_name"].lower().split():
                if term in diagnosis.lower() and len(term) > 3:  # Avoid matching short words
                    score += 0.2
                    break
    
    return min(1.0, score)