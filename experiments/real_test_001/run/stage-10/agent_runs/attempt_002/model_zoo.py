import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class SecurityModel(nn.Module):
    """A wrapper for GPT-2 to facilitate security experiments."""
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
    def forward(self, input_ids, labels=None):
        outputs = self.model(input_ids, labels=labels)
        return outputs.logits, outputs.loss

    def generate(self, input_ids, max_length=16):
        return self.model.generate(input_ids, max_new_tokens=max_length, pad_token_id=self.model.config.eos_token_id)

class SmoothLLMWrapper(nn.Module):
    """Implements SmoothLLM: ensemble defense via input perturbation."""
    def __init__(self, base_model, config, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.tokenizer = tokenizer

    def forward(self, input_ids):
        # input_ids: [B, S]
        B, S = input_ids.shape
        N = self.config.smooth_llm_samples
        
        # Create ensemble
        all_logits = []
        for _ in range(N):
            perturbed_ids = input_ids.clone()
            # Randomly flip 1 token per sequence
            for i in range(B):
                idx = torch.randint(0, S, (1,))
                perturbed_ids[i, idx] = torch.randint(0, self.tokenizer.vocab_size, (1,))
            
            logits, _ = self.base_model(perturbed_ids)
            all_logits.append(logits.unsqueeze(1)) # [B, 1, S, V]
            
        avg_logits = torch.cat(all_logits, dim=1).mean(dim=1)
        return avg_logits, None

class PerplexityFilterModel(nn.Module):
    """Rejects high-perplexity inputs as potential adversarial attacks."""
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.ref_model = base_model # Use same model for PPL check
        
    def forward(self, input_ids):
        # Compute PPL
        with torch.no_grad():
            outputs = self.ref_model.model(input_ids, labels=input_ids)
            loss = outputs.loss
            ppl = loss # Using log-ppl (loss) directly
            
        if ppl > self.config.ppl_threshold:
            # Refuse: return a distribution concentrated on a 'refusal' token
            B, S = input_ids.shape
            V = self.base_model.model.config.vocab_size
            refusal_logits = torch.zeros((B, S, V), device=input_ids.device)
            refusal_logits[:, :, self.config.refusal_token_id] = 10.0
            return refusal_logits, torch.tensor(0.0, device=input_ids.device)
        
        return self.base_model(input_ids)