import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import apply_adversarial_perturbation

class BaseTrainer:
    def __init__(self, model, config, tokenizer, safety_scorer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.safety_scorer = safety_scorer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.config.device)
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        results = {}
        for strength in self.config.adv_strengths:
            asr, harmfulness = [], []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.config.device)
                    adv_ids = apply_adversarial_perturbation(input_ids, self.tokenizer, strength=strength)
                    gen_ids = self.model.generate(adv_ids, max_length=12)
                    
                    scores = self.safety_scorer(gen_ids)
                    harmfulness.extend(scores.cpu().tolist())
                    asr.extend((scores > 0.3).float().cpu().tolist())
            results[f'ASR_at_{strength}'] = np.mean(asr)
        return results

class AdversarialTrainer(BaseTrainer):
    """Defense via multi-strength adversarial training and KL-regularization."""
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.config.device)
            # Create adversarial examples with varying intensity
            adv_ids = apply_adversarial_perturbation(input_ids, self.tokenizer, strength=0.15)
            
            self.optimizer.zero_grad()
            out_clean = self.model(input_ids, labels=input_ids)
            out_adv = self.model(adv_ids, labels=adv_ids)
            
            # KL divergence to keep adv predictions close to clean predictions (stability)
            p_clean = F.softmax(out_clean.logits, dim=-1)
            log_p_adv = F.log_softmax(out_adv.logits, dim=-1)
            kl_loss = F.kl_div(log_p_adv, p_clean, reduction='batchmean')
            
            loss = 0.4 * out_clean.loss + 0.4 * out_adv.loss + 0.2 * kl_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

class RLUFTrainer(BaseTrainer):
    """Reinforcement Learning from Unsafe Feedback with per-sample rewards."""
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.config.device)
            with torch.no_grad():
                gen_ids = self.model.generate(input_ids, max_length=10)
            
            rewards = 1.0 - self.safety_scorer(gen_ids) # Higher is safer
            
            self.optimizer.zero_grad()
            outputs = self.model(gen_ids, labels=gen_ids)
            logits = outputs.logits
            
            # Manual calculation of per-sample NLL to apply sample-specific rewards
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = gen_ids[..., 1:].contiguous()
            
            # CrossEntropy with reduction=None to get (Batch, Seq) losses
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_sample_loss = per_token_loss.view(gen_ids.size(0), -1).mean(dim=1)
            
            # Policy Gradient: weight each sample's loss by its (negative) reward
            # We minimize -Reward * LogProb, which is Reward * NLL
            policy_loss = (per_sample_loss * (2.0 - rewards)).mean()
            
            policy_loss.backward()
            self.optimizer.step()
            total_loss += policy_loss.item()
        return total_loss / len(loader)