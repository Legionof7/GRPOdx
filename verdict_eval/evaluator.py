try:
    from verdict import Pipeline  
    from verdict.common.judge import NumericJudgeUnit
    from verdict.schema import Schema
except ImportError:
    # If verdict is not installed, provide informative error
    raise ImportError(
        "Verdict package is not installed. Please install it using: "
        "pip install verdict or run install_verdict.py"
    )
from typing import Dict, Optional

def create_verdict_evaluator():
    """Create a Verdict pipeline for evaluating medical diagnoses."""
    evaluator = NumericJudgeUnit(name='DiagnosisEvaluator', min_value=0.0, max_value=1.0).prompt("""
        You are a medical evaluation system. Your task is to evaluate a doctor's diagnosis 
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
        
        Correct Diagnosis: {source.correct_disease}
        Symptoms Present: {source.symptoms_present}
        Symptoms Absent: {source.symptoms_absent}
        
        Doctor's Response:
        {source.response}
        
        Evaluate and provide only a single number from 0.0 to 1.0:
    """).via(policy_or_name='gpt-4o-mini', retries=2, temperature=0.1)
    
    return Pipeline() >> evaluator

def evaluate_diagnosis(response: str, disease_info: Dict, verbose: bool = True) -> float:
    """
    Evaluate a diagnosis using Verdict.
    
    Args:
        response: The doctor's full response
        disease_info: Dictionary containing disease information
        verbose: Whether to print evaluation details
    
    Returns:
        Evaluation score between 0.0 and 1.0
    """
    evaluator = create_verdict_evaluator()
    
    # Get symptoms present and absent
    symptoms_present = [s for s, v in disease_info['symptoms'].items() if v]
    symptoms_absent = [s for s, v in disease_info['symptoms'].items() if not v]
    
    # Create input schema for verdict
    input_data = Schema.of(
        correct_disease=disease_info['disease_name'],
        symptoms_present=', '.join(symptoms_present),
        symptoms_absent=', '.join(symptoms_absent),
        response=response
    )
    
    if verbose:
        print(f"Evaluating diagnosis with Verdict...")
    
    # Run the evaluation
    response, _ = evaluator.run(input_data, max_workers=1)
    
    # Extract the score
    score = response['DiagnosisEvaluator_root.block.unit_score']
    
    if verbose:
        print(f"Verdict score: {score:.2f}")
    
    return score