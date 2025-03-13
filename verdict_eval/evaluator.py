import sys
import os
import subprocess

# Try to import Verdict, install it if not available
try:
    from verdict import Pipeline  
    from verdict.common.judge import NumericJudgeUnit
    from verdict.schema import Schema
except ImportError:
    # If Verdict is not installed, try to install it
    print("Verdict not found. Installing...")
    try:
        # Use subprocess with a timeout to prevent hanging
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "verdict"],
            timeout=60
        )
        # Import after installation
        from verdict import Pipeline  
        from verdict.common.judge import NumericJudgeUnit
        from verdict.schema import Schema
        print("Successfully installed and imported Verdict")
    except subprocess.TimeoutExpired:
        print("Verdict installation timed out, using fallback")
        # Define dummy classes to prevent import errors
        class DummySchema:
            @staticmethod
            def of(**kwargs):
                return kwargs
                
        class DummyPipeline:
            def __init__(self):
                pass
            def __rshift__(self, other):
                return self
            def run(self, *args, **kwargs):
                return {"DiagnosisEvaluator_root.block.unit_score": 0.5}, None
                
        class DummyNumericJudgeUnit:
            def __init__(self, name='', min_value=0.0, max_value=1.0):
                self.name = name
            def prompt(self, template):
                return self
            def via(self, policy_or_name='', retries=0, temperature=0.0):
                return self
                
        # Assign dummy classes
        Pipeline = DummyPipeline
        NumericJudgeUnit = DummyNumericJudgeUnit
        Schema = DummySchema
        print("Using dummy Verdict classes for fallback")
    except Exception as e:
        # Still couldn't install, use fallback instead of raising error
        print(f"Failed to install Verdict: {e}")
        # Define dummy classes as above
        class DummySchema:
            @staticmethod
            def of(**kwargs):
                return kwargs
                
        class DummyPipeline:
            def __init__(self):
                pass
            def __rshift__(self, other):
                return self
            def run(self, *args, **kwargs):
                return {"DiagnosisEvaluator_root.block.unit_score": 0.5}, None
                
        class DummyNumericJudgeUnit:
            def __init__(self, name='', min_value=0.0, max_value=1.0):
                self.name = name
            def prompt(self, template):
                return self
            def via(self, policy_or_name='', retries=0, temperature=0.0):
                return self
                
        # Assign dummy classes
        Pipeline = DummyPipeline
        NumericJudgeUnit = DummyNumericJudgeUnit
        Schema = DummySchema
        print("Using dummy Verdict classes after install failure")
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

def rule_based_evaluation(response: str, disease_info: Dict, verbose: bool = False) -> float:
    """Rule-based evaluation as fallback when Verdict is unavailable."""
    import re
    
    # Extract diagnosis
    diagnosis = None
    diagnosis_match = re.search(r"final diagnosis:\s*([^.\n]+)", response.lower())
    if diagnosis_match:
        diagnosis = diagnosis_match.group(1).strip()
    else:
        # Try to extract from the <answer> tag
        answer_match = re.search(r"<answer>\s*([^<]+)", response)
        if answer_match:
            diagnosis = answer_match.group(1).strip()
    
    # Check for reasoning
    has_reasoning = False
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    if reasoning_match:
        has_reasoning = True
    
    # Calculate score components
    base_score = 0.1
    format_score = 0.0
    diagnosis_score = 0.0
    
    # Format scoring
    if "<reasoning>" in response and "</reasoning>" in response:
        format_score += 0.2
    if "<answer>" in response and "</answer>" in response:
        format_score += 0.1
    if "final diagnosis:" in response.lower():
        format_score += 0.1
    
    # Diagnosis scoring
    if diagnosis:
        if disease_info["disease_name"].lower() in diagnosis.lower():
            diagnosis_score = 0.6
        else:
            # Check for partial matches
            for term in disease_info["disease_name"].lower().split():
                if term in diagnosis.lower() and len(term) > 3:  # Avoid matching short words
                    diagnosis_score = 0.2
                    break
    
    # Calculate final score
    total_score = min(1.0, base_score + format_score + diagnosis_score)
    
    if verbose:
        print(f"Rule-based evaluation:")
        print(f"  Diagnosis: {diagnosis or 'None'}")
        print(f"  Has reasoning: {has_reasoning}")
        print(f"  Base score: {base_score:.2f}")
        print(f"  Format score: {format_score:.2f}")
        print(f"  Diagnosis score: {diagnosis_score:.2f}")
        print(f"  Total score: {total_score:.2f}")
    
    return total_score

def evaluate_diagnosis(response: str, disease_info: Dict, verbose: bool = False) -> float:
    """
    Evaluate a diagnosis using Verdict if available, with fallback to rule-based scoring.
    
    Args:
        response: The doctor's full response
        disease_info: Dictionary containing disease information
        verbose: Whether to print evaluation details
    
    Returns:
        Evaluation score between 0.0 and 1.0
    """
    try:
        if verbose:
            print("Attempting to use Verdict for evaluation...")
            
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
        
        # Run the evaluation with a timeout to prevent hanging
        import threading
        import time
        
        result = [None]
        error = [None]
        
        def run_evaluation():
            try:
                verdict_response, _ = evaluator.run(input_data, max_workers=1)
                result[0] = verdict_response
            except Exception as e:
                error[0] = e
        
        # Run the evaluation in a thread with a timeout
        thread = threading.Thread(target=run_evaluation)
        thread.daemon = True
        thread.start()
        
        # Wait for the thread to complete or timeout
        thread.join(10.0)  # 10 second timeout
        
        if thread.is_alive() or error[0] is not None or result[0] is None:
            # Timeout or error occurred
            if verbose:
                print("Verdict evaluation timed out or failed")
            # Use fallback evaluation
            return rule_based_evaluation(response, disease_info, verbose)
            
        # Extract the score
        score = result[0]['DiagnosisEvaluator_root.block.unit_score']
        
        if verbose:
            print(f"Verdict evaluation score: {score:.2f}")
            
        return score
            
    except Exception as e:
        if verbose:
            print(f"Verdict evaluation failed: {e}")
            print("Using fallback rule-based scoring...")
            
        # Fallback to rule-based scoring
        return rule_based_evaluation(response, disease_info, verbose)