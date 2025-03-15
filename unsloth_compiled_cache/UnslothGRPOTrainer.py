"""
Modified version of GRPOTrainer for use with Unsloth.
"""

from trl import GRPOTrainer
from trl.trainer.grpo_trainer import pad
import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
import os
from contextlib import nullcontext

class UnslothGRPOTrainer(GRPOTrainer):
    """
    Custom GRPOTrainer that handles Unsloth and vLLM correctly.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_loaded_step = -1
        self.llm = None
        
    def _move_model_to_vllm(self):
        """Move model to vLLM when needed."""
        from vllm import SamplingParams
        
        # Create sampling params
        self.sampling_params = SamplingParams(
            temperature=self.args.temperature,
            max_tokens=self.max_completion_length,
            top_p=0.95,
        )
        
        # Get vLLM engine from model
        self.llm = self.accelerator.unwrap_model(self.model).model.model
    
    def multi_turn_generation(self, prompt, model, tokenizer, generation_config, max_new_tokens=50):
        """
        Default implementation that just runs text generation once.
        Override this to implement multi-turn interactions.
        """
        outputs = self.llm.generate(
            [prompt],
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        completion_ids = outputs[0].outputs[0].token_ids
        return completion_ids, 0.0  # No reward by default
    
    def _get_per_token_logps(self, model, prompt_completion_ids, attention_mask, logits_to_keep):
        """Get per-token log probabilities."""
        outputs = model(
            input_ids=prompt_completion_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits[:, :-1]
        
        # Get next tokens for each position
        next_tokens = prompt_completion_ids[:, 1:]
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs of next tokens
        next_token_logps = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Apply attention mask from padding
        mask = attention_mask[:, 1:].bool()
        per_token_logps = torch.full_like(next_token_logps, -1e8)
        per_token_logps = per_token_logps.masked_scatter_(mask, next_token_logps.masked_select(mask))
        
        # Mask prefix
        prefix_length = prompt_completion_ids.size(1) - logits_to_keep
        per_token_logps[:, :prefix_length] = 0
        
        return per_token_logps