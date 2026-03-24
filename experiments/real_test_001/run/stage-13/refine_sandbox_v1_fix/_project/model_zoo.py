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

    def generate(self, input_ids, max_new_tokens=16):
        return self.model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            pad_token_id=self.model.config.eos_token_id,
            do_sample=False
        )

class LSTMLM(nn.Module):
    """Baseline LSTM Language Model."""
    def __init__(self, vocab_size=50257, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.config = type('Config', (), {'eos_token_id': 50256})()

    def forward(self, input_ids, labels=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.linear(lstm_out)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type('Output', (), {'loss': loss, 'logits': logits})()

    def generate(self, input_ids, max_new_tokens=16, **kwargs):
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

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

    def generate(self, input_ids, max_new_tokens=16):
        B = input_ids.shape[0]
        N = self.config.smooth_llm_samples
        final_outputs = []

        for i in range(B):
            single_input = input_ids[i:i+1].repeat(N, 1)
            perturbed = apply_adversarial_perturbation(single_input, self.tokenizer, strength=0.1)
            candidates = self.base_model.generate(perturbed, max_new_tokens=max_new_tokens)
            with torch.no_grad():
                scores = self.safety_scorer(candidates)
                best_idx = torch.argmin(scores)
                final_outputs.append(candidates[best_idx : best_idx+1])
        return torch.cat(final_outputs, dim=0)

class PerplexityFilterModel(nn.Module):
    """Defense via input filtering based on perplexity."""
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
    def forward(self, input_ids, labels=None):
        return self.base_model(input_ids, labels=labels)

    def generate(self, input_ids, max_new_tokens=16):
        B = input_ids.shape[0]
        final_gen = []
        for i in range(B):
            single_input = input_ids[i:i+1]
            with torch.no_grad():
                outputs = self.base_model(single_input, labels=single_input)
                ppl = torch.exp(outputs.loss)
            
            if ppl.item() > self.config.ppl_threshold:
                # Refusal response
                refusal = torch.full((1, max_new_tokens), self.config.refusal_token_id, device=input_ids.device)
                final_gen.append(torch.cat([single_input, refusal], dim=1))
            else:
                final_gen.append(self.base_model.generate(single_input, max_new_tokens=max_new_tokens))
        return torch.cat(final_gen, dim=0)