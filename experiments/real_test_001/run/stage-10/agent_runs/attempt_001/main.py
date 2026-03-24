"""
Dataset: WikiText-103 subset (pre-cached)
Model: GPT-2 Small (pre-trained, transformers)
Training: Fine-tuning for 1 epoch per condition with specialized losses
Evaluation: Attack Success Rate (ASR), Harmfulness, Perplexity (PPL)
"""
import torch
import numpy as np
import time
import json
from config import ExperimentConfig, HYPERPARAMETERS
from data_utils import get_wiki_loaders, get_harmfulness_scorer
from model_zoo import SecurityModel, SmoothLLMWrapper, PerplexityFilterModel
from engine import BaseTrainer, AdversarialTrainer, RLUFTrainer
from experiment_harness import ExperimentHarness

def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("METRIC_DEF: ASR | direction=lower | Attack Success Rate (0-1)")
    print("METRIC_DEF: Harmfulness | direction=lower | Mean harmfulness score (0-1)")
    print("METRIC_DEF: PPL | direction=lower | Model Perplexity on clean text")
    print("REGISTERED_CONDITIONS: Baseline, AdvTrain, RLUF, SmoothLLM, PPLFilter")
    
    harness = ExperimentHarness(time_budget=600)
    config = ExperimentConfig()
    start_time = time.time()
    
    train_loader, test_loader, tokenizer = get_wiki_loaders(config)
    safety_scorer = get_harmfulness_scorer(tokenizer)
    
    conditions = ['Baseline', 'AdvTrain', 'RLUF', 'SmoothLLM', 'PPLFilter']
    all_results = {c: [] for c in conditions}
    
    print(f"TIME_ESTIMATE: {len(conditions) * len(config.seeds) * 20}s")

    for condition in conditions:
        print(f"Running condition: {condition}")
        for seed in config.seeds:
            if harness.should_stop():
                break
                
            set_all_seeds(seed)
            model = SecurityModel().to(config.device)
            
            # Setup Trainer and Model Wrapper
            if condition == 'Baseline':
                trainer = BaseTrainer(model, config, tokenizer, safety_scorer)
            elif condition == 'AdvTrain':
                trainer = AdversarialTrainer(model, config, tokenizer, safety_scorer)
            elif condition == 'RLUF':
                trainer = RLUFTrainer(model, config, tokenizer, safety_scorer)
            elif condition == 'SmoothLLM':
                wrapped_model = SmoothLLMWrapper(model, config, tokenizer)
                trainer = BaseTrainer(wrapped_model, config, tokenizer, safety_scorer)
            elif condition == 'PPLFilter':
                wrapped_model = PerplexityFilterModel(model, config)
                trainer = BaseTrainer(wrapped_model, config, tokenizer, safety_scorer)
            
            # Train (except for inference-only defenses which we still 'fit' for consistency)
            try:
                loss = trainer.train_epoch(train_loader)
                if not harness.check_value(loss, 'loss'):
                    print(f"FAIL: NaN detected in {condition} seed {seed}")
                    continue
                
                metrics = trainer.evaluate(test_loader)
                all_results[condition].append(metrics)
                print(f"condition={condition} seed={seed} ASR: {metrics['ASR']:.4f} Harm: {metrics['Harmfulness']:.4f} PPL: {metrics['PPL']:.2f}")
            except Exception as e:
                print(f"CONDITION_FAILED: {condition} seed {seed} Error: {e}")
                continue

        # Aggregate results for condition
        if all_results[condition]:
            m_asr = np.mean([r['ASR'] for r in all_results[condition]])
            s_asr = np.std([r['ASR'] for r in all_results[condition]])
            m_harm = np.mean([r['Harmfulness'] for r in all_results[condition]])
            m_ppl = np.mean([r['PPL'] for r in all_results[condition]])
            
            harness.report_metric(f'{condition}_asr_mean', m_asr)
            print(f"condition={condition} ASR_mean: {m_asr:.4f} ASR_std: {s_asr:.4f}")
            print(f"condition={condition} Harmfulness_mean: {m_harm:.4f}")
            print(f"condition={condition} PPL_mean: {m_ppl:.2f}")

    # Final summary
    summary_parts = [f"{c}={np.mean([r['ASR'] for r in all_results[c]]):.4f}" for c in conditions if all_results[c]]
    print("SUMMARY: " + ", ".join(summary_parts))
    
    results_json = {
        'hyperparameters': HYPERPARAMETERS,
        'metrics': all_results
    }
    with open('results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
        
    harness.finalize()

if __name__ == "__main__":
    main()