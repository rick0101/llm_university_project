import torch
import torch.nn as nn
import torch.nn.functional as F
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

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type('Output', (), {'logits': logits, 'loss': loss})

    def generate(self, input_ids, max_new_tokens=16):
        self.eval()
        curr_ids = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(curr_ids)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
        return curr_ids

class SmoothLLMWrapper(nn.Module):
    def __init__(self, base_model, config, tokenizer, safety_scorer):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.tokenizer = tokenizer
        self.safety_scorer = safety_scorer

    def generate(self, input_ids, max_length=16):
        B = input_ids.shape[0]
        N = self.config.smooth_llm_samples
        final_outputs = []
        consistency_scores = []

        for i in range(B):
            single_input = input_ids[i:i+1].repeat(N, 1)
            perturbed = apply_adversarial_perturbation(single_input, self.tokenizer, strength=0.1)
            candidates = self.base_model.generate(perturbed, max_length=max_length)
            
            with torch.no_grad():
                harm_scores, _ = self.safety_scorer(candidates)
                best_idx = torch.argmin(harm_scores)
                final_outputs.append(candidates[best_idx : best_idx+1])
                
                # Consistency: Mean overlap of token IDs with the chosen best
                matches = (candidates == candidates[best_idx]).float().mean()
                consistency_scores.append(matches.item())
        
        self.last_consistency = np.mean(consistency_scores)
        return torch.cat(final_outputs, dim=0)

class PerplexityFilterModel(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
    def generate(self, input_ids, max_length=16):
        B, L = input_ids.shape
        with torch.no_grad():
            outputs = self.base_model(input_ids, labels=input_ids)
            ppl = torch.exp(outputs.loss)
            
        if ppl.item() > self.config.ppl_threshold:
            # Return a refusal sequence (EOS tokens)
            return torch.full((B, L + max_length), self.config.refusal_token_id, device=input_ids.device)
        return self.base_model.generate(input_ids, max_length=max_length)