import torch
import torch.nn as nn
import time
import numpy as np

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
            logits, loss = self.model(input_ids, labels=input_ids)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        asr = []
        harmfulness = []
        ppls = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                # Test with adversarial input
                from data_utils import apply_adversarial_perturbation
                adv_ids = apply_adversarial_perturbation(input_ids, self.tokenizer, strength=0.1)
                
                # Generate responses
                if hasattr(self.model, 'base_model'):
                    gen_ids = self.model.base_model.generate(adv_ids)
                else:
                    gen_ids = self.model.generate(adv_ids)
                
                # Compute metrics
                harm_scores = self.safety_scorer(gen_ids)
                harmfulness.extend(harm_scores.tolist())
                asr.extend((harm_scores > 0.3).float().tolist())
                
                # PPL on clean input
                outputs = self.model.model(input_ids, labels=input_ids) if not hasattr(self.model, 'base_model') else self.model.base_model.model(input_ids, labels=input_ids)
                ppls.append(torch.exp(outputs.loss).item())

        return {
            'ASR': np.mean(asr),
            'Harmfulness': np.mean(harmfulness),
            'PPL': np.mean(ppls)
        }

class AdversarialTrainer(BaseTrainer):
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.config.device)
            
            # Simple FGSM-like discrete perturbation (random in this simplified version)
            from data_utils import apply_adversarial_perturbation
            adv_ids = apply_adversarial_perturbation(input_ids, self.tokenizer, strength=0.05)
            
            self.optimizer.zero_grad()
            _, loss_clean = self.model(input_ids, labels=input_ids)
            _, loss_adv = self.model(adv_ids, labels=adv_ids)
            
            loss = 0.5 * (loss_clean + loss_adv)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

class RLUFTrainer(BaseTrainer):
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.config.device)
            
            # Generate and calculate reward
            outputs = self.model.generate(input_ids, max_length=10)
            rewards = -1.0 * self.safety_scorer(outputs) # Reward is negative harmfulness
            
            self.optimizer.zero_grad()
            logits, lm_loss = self.model(outputs, labels=outputs)
            
            # Policy gradient: -log_prob * reward
            # Simplified: use cross-entropy loss as -log_prob proxy
            policy_loss = (lm_loss * rewards.mean().detach())
            
            total_l = policy_loss + self.config.safety_reward_beta * lm_loss
            total_l.backward()
            self.optimizer.step()
            total_loss += total_l.item()
        return total_loss / len(loader)