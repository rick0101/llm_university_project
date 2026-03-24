import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def get_wiki_loaders(config):
    """Loads a subset of WikiText-103 for efficient security probing."""
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='test', cache_dir='/workspace/data/hf')
    
    # Filter very short lines
    dataset = dataset.filter(lambda x: len(x['text']) > 50)
    
    # Take small subsets for the experiment budget
    train_data = dataset.select(range(config.train_samples))
    test_data = dataset.select(range(config.train_samples, config.train_samples + config.eval_samples))
    
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=config.max_seq_len, return_tensors='pt')

    train_tokenized = train_data.map(tokenize, batched=True, remove_columns=['text'])
    test_tokenized = test_data.map(tokenize, batched=True, remove_columns=['text'])
    
    train_tokenized.set_format('torch')
    test_tokenized.set_format('torch')
    
    train_loader = torch.utils.data.DataLoader(train_tokenized, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_tokenized, batch_size=config.batch_size)
    
    return train_loader, test_loader, tokenizer

def apply_adversarial_perturbation(input_ids, tokenizer, strength=0.1):
    """Perturbs input IDs by randomly swapping tokens with random vocabulary IDs."""
    perturbed = input_ids.clone()
    batch_size, seq_len = input_ids.shape
    num_to_perturb = max(1, int(seq_len * strength))
    
    for i in range(batch_size):
        indices = np.random.choice(seq_len, num_to_perturb, replace=False)
        for idx in indices:
            if perturbed[i, idx] != tokenizer.eos_token_id:
                perturbed[i, idx] = np.random.randint(0, tokenizer.vocab_size)
    return perturbed

def get_harmfulness_scorer(tokenizer):
    """
    Simulates a harmfulness scorer. 
    In a real scenario, this would be a trained classifier.
    Here, we detect 'harmful' tokens based on a synthetic list.
    """
    # Synthetic 'harmful' token IDs for GPT-2 tokenizer
    harmful_keywords = ["bad", "hurt", "kill", "attack", "dangerous", "exploit"]
    harmful_ids = [tokenizer.encode(w, add_prefix_space=True)[0] for w in harmful_keywords]
    
    def score(input_ids):
        # input_ids: [B, S]
        scores = torch.zeros(input_ids.shape[0], device=input_ids.device)
        for i in range(input_ids.shape[0]):
            count = 0
            for tid in input_ids[i]:
                if tid.item() in harmful_ids:
                    count += 1
            scores[i] = min(1.0, count / 2.0) # Normalize
        return scores
    
    return score