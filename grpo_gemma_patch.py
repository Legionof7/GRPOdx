"""
GRPO patch for Gemma models to fix dimension mismatch issues.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def patch_grpo_for_gemma():
    """
    Apply monkey patches to fix dimension mismatch in GRPO with Gemma models.
    Should be called before initializing the GRPOTrainer.
    """
    # First, monkey patch the matmul in torch.func implementation for GRPO
    try:
        from torch._functorch.eager_transforms import _wrap_functional_tensor

        original_matmul = torch.matmul
        
        def safe_matmul(input, other, *, out=None):
            """Safe matmul that handles dimension mismatches by reshaping."""
            try:
                return original_matmul(input, other, out=out)
            except RuntimeError as e:
                # Check if it's the specific dimension mismatch we're trying to fix
                if "a and b must have same reduction dim" in str(e):
                    # Get the dimensions
                    if hasattr(input, "shape") and hasattr(other, "shape"):
                        # If first tensor is (1, s0, 262144) and second is (1152, 262144)
                        # We need to reshape for proper broadcasting
                        if len(input.shape) == 3 and len(other.shape) == 2 and input.shape[2] == other.shape[1]:
                            # Reshape to allow proper broadcasting
                            reshaped_input = input.reshape(-1, input.shape[2])
                            result = original_matmul(reshaped_input, other, out=out)
                            # Reshape back to expected output shape
                            return result.reshape(input.shape[0], input.shape[1], other.shape[0])
                # For any other errors or if reshaping failed, reraise the original exception
                raise e
        
        # Apply the patch
        torch.matmul = safe_matmul
        
        print("Successfully applied GRPO patch for Gemma models")
    except Exception as e:
        print(f"Warning: Couldn't apply GRPO patch: {e}")
        print("You might still encounter dimension mismatch errors.")

def ensure_eager_attention(model_path):
    """
    Ensure the model is loaded with eager attention implementation.
    """
    if "gemma" in model_path.lower():
        os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
        print("Set environment to use eager attention implementation for Gemma")
    return model_path