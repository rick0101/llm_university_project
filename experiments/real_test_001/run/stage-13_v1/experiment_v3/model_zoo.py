import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from data_utils import apply_adversarial_perturbation

class SecurityModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels, return_dict=True)

    def generate(self, input_ids, max_length=16):
        return self.model.generate(
            input_ids, 
            max_new_tokens=max_length, 
            pad_token_id=self.model.config.eos_token_id,
            do_sample=False
        )

class SmoothLLMWrapper(nn.Module):
    """Defense via ensemble generation on perturbed inputs."""
    def __init__(self, base_model, config, tokenizer, safety_scorer):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.tokenizer = tokenizer
        self.safety_scorer = safety_scorer

    def forward(self, input_ids, labels=None):
        return self.base_model(input_ids, labels=labels)

    def generate(self, input_ids, max_length=16):
        B = input_ids.shape[0]
        N = self.config.smooth_llm_samples
        final_outputs = []

        for i in range(B):
            # Create N perturbed versions of a single input
            single_input = input_ids[i:i+1].repeat(N, 1)
            perturbed = apply_adversarial_perturbation(single_input, self.tokenizer, strength=0.1)
            
            # Generate N candidate completions
            candidates = self.base_model.generate(perturbed, max_length=max_length)
            
            # Score candidates and pick the one with the lowest harmfulness
            with torch.no_grad():
                scores = self.safety_scorer(candidates)
                best_idx = torch.argmin(scores)
                final_outputs.append(candidates[best_idx : best_idx+1])
        
        return torch.cat(final_outputs, dim=0)

class PerplexityFilterModel(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
    def generate(self, input_ids, max_length=16):
        B = input_ids.shape[0]
        with torch.no_grad():
            outputs = self.base_model(input_ids, labels=input_ids)
            ppl = torch.exp(outputs.loss)
            
        if ppl.item() > self.config.ppl_threshold:
            return torch.full((B, 1), self.config.refusal_token_id, device=input_ids.device)
        return self.base_model.generate(input_ids, max_length=max_length)