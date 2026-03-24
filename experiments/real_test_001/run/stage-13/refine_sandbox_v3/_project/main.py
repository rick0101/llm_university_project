import torch
import random
import numpy as np
from config import ExperimentConfig
from data_utils import get_wiki_loaders, get_harmfulness_scorer
from model_zoo import SecurityModel, LSTMLM, SmoothLLMWrapper, PerplexityFilterModel
from engine import BaseTrainer, AdversarialTrainer, RLUFTrainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_condition(condition_name, config, train_loader, test_loader, tokenizer, safety_scorer):
    print(f"Running condition: {condition_name}")
    set_seed(config.seeds[0])
    
    # Setup model and trainer based on condition
    if condition_name == "GPT-2 Small":
        model = SecurityModel().to(config.device)
        trainer = BaseTrainer(model, config, tokenizer, safety_scorer)
    elif condition_name == "LSTM LM":
        model = LSTMLM(vocab_size=tokenizer.vocab_size).to(config.device)
        trainer = BaseTrainer(model, config, tokenizer, safety_scorer)
    elif condition_name == "Adversarial Training with Safety Constraints":
        model = SecurityModel().to(config.device)
        trainer = AdversarialTrainer(model, config, tokenizer, safety_scorer)
    elif condition_name == "RLUF":
        model = SecurityModel().to(config.device)
        trainer = RLUFTrainer(model, config, tokenizer, safety_scorer)
    elif condition_name == "SmoothLLM Defense":
        base_m = SecurityModel().to(config.device)
        model = SmoothLLMWrapper(base_m, config, tokenizer, safety_scorer).to(config.device)
        trainer = BaseTrainer(model, config, tokenizer, safety_scorer)
    elif condition_name == "Perplexity-based Input Filtering":
        base_m = SecurityModel().to(config.device)
        model = PerplexityFilterModel(base_m, config, tokenizer).to(config.device)
        trainer = BaseTrainer(model, config, tokenizer, safety_scorer)
    elif condition_name == "Removal of system-level safety prompts":
        config.use_safety_prompt = False
        # Reload loaders without prompt
        train_l, test_l, _ = get_wiki_loaders(config)
        model = SecurityModel().to(config.device)
        trainer = BaseTrainer(model, config, tokenizer, safety_scorer)
        trainer.train_epoch(train_l)
        results = trainer.evaluate(test_l)
        config.use_safety_prompt = True # Reset
        print(f"condition={condition_name} {results}")
        return results
    elif condition_name == "Varying the strength of adversarial perturbations":
        # This condition tests the model under a high-perturbation regime specifically
        old_strengths = config.adv_strengths
        config.adv_strengths = [0.5, 0.8]
        model = SecurityModel().to(config.device)
        trainer = BaseTrainer(model, config, tokenizer, safety_scorer)
        trainer.train_epoch(train_loader)
        results = trainer.evaluate(test_loader)
        config.adv_strengths = old_strengths # Reset
        print(f"condition={condition_name} {results}")
        return results
    else:
        return {}

    # Standard train/eval for other conditions
    trainer.train_epoch(train_loader)
    results = trainer.evaluate(test_loader)
    print(f"condition={condition_name} {results}")
    return results

def main():
    config = ExperimentConfig()
    train_loader, test_loader, tokenizer = get_wiki_loaders(config)
    safety_scorer = get_harmfulness_scorer(tokenizer)
    
    conditions = [
        "GPT-2 Small",
        "LSTM LM",
        "Adversarial Training with Safety Constraints",
        "RLUF",
        "SmoothLLM Defense",
        "Perplexity-based Input Filtering",
        "Removal of system-level safety prompts",
        "Varying the strength of adversarial perturbations"
    ]
    
    summary = {}
    for cond in conditions:
        summary[cond] = run_condition(cond, config, train_loader, test_loader, tokenizer, safety_scorer)

    print("\n" + "="*50)
    print("FINAL SUMMARY COMPARISON")
    print(f"{'Condition':<45} | Robustness (1-ASR)")
    print("-" * 70)
    for cond, res in summary.items():
        if res:
            print(f"{cond:<45} | {res['primary_metric']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()