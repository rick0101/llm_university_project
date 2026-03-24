import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from data_utils import apply_adversarial_perturbation

class SecurityModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
    def forward(self, input_ids, labels=None, attention_mask=None):
        return self.model(input_ids, labels=labels, attention_mask=attention_mask, return_dict=True)

    def generate(self, input_ids, max_length=16):
        return self.model.generate(
            input_ids, 
            max_new_tokens=max_length, 
            pad_token_id=self.model.config.eos_token_id,
            do_sample=False
        )

class LSTMLM(nn.Module):
    """LSTM-based Baseline Language Model."""
    def __init__(self, vocab_size=50257, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.config = type('Config', (), {'eos_token_id': 50256})()

    def forward(self, input_ids, labels=None, attention_mask=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.lm_head(lstm_out)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type('Output', (), {'loss': loss, 'logits': logits})()

    def generate(self, input_ids, max_new_tokens=16, pad_token_id=50256, **kwargs):
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(generated)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if (next_token == pad_token_id).all(): break
        return generated

class SmoothLLMWrapper(nn.Module):
    """Defense via ensemble generation on perturbed inputs."""
    def __init__(self, base_model, config, tokenizer, safety_scorer):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.tokenizer = tokenizer
        self.safety_scorer = safety_scorer

    def forward(self, input_ids, labels=None, **kwargs):
        return self.base_model(input_ids, labels=labels, **kwargs)

    def generate(self, input_ids, max_length=16):
        B = input_ids.shape[0]
        N = self.config.smooth_llm_samples
        final_outputs = []

        for i in range(B):
            single_input = input_ids[i:i+1].repeat(N, 1)
            perturbed = apply_adversarial_perturbation(single_input, self.tokenizer, strength=0.1)
            candidates = self.base_model.generate(perturbed, max_length=max_length)
            with torch.no_grad():
                scores = self.safety_scorer(candidates)
                best_idx = torch.argmin(scores)
                final_outputs.append(candidates[best_idx : best_idx+1])
        return torch.cat(final_outputs, dim=0)

class PerplexityFilterModel(nn.Module):
    """Defense via perplexity-based input filtering."""
    def __init__(self, base_model, config, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.tokenizer = tokenizer
        
    def generate(self, input_ids, max_length=16):
        B = input_ids.shape[0]
        final_gen = []
        for i in range(B):
            sample_input = input_ids[i:i+1]
            with torch.no_grad():
                outputs = self.base_model(sample_input, labels=sample_input)
                ppl = torch.exp(outputs.loss)
            
            if ppl.item() > self.config.ppl_threshold:
                # Return refusal
                refusal_ids = self.tokenizer.encode(self.config.refusal_string, return_tensors='pt').to(input_ids.device)
                final_gen.append(refusal_ids)
            else:
                final_gen.append(self.base_model.generate(sample_input, max_length=max_length))
        
        # Pad to match lengths if necessary
        max_len = max(x.shape[1] for x in final_gen)
        padded = [torch.cat([x, torch.full((1, max_len - x.shape[1]), self.config.refusal_token_id, device=input_ids.device)], dim=1) for x in final_gen]
        return torch.cat(padded, dim=0)