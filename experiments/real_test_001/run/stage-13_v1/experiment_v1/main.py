Here is the corrected code. I have addressed the `ModuleNotFoundError` by adding a fallback for the `datasets` library, implemented RL stability measures (gradient clipping, NaN guards, and reward clipping), fixed the `safety_scorer` initialization, and ensured compatibility with modern NumPy/Torch patterns.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import AutoTokenizer, GPT2LMHeadModel

# --- config.py ---
class ExperimentConfig:
    def __init__(self):
        self.lr = 1e-4  # Lowered for RL stability
        self.batch_size = 4 
        self.max_seq_len = 32
        self.num_epochs = 1
        self.train_samples = 16
        self.eval_samples = 12
        self.adv_strengths = [0.0, 0.1, 0.2, 0.4] 
        self.smooth_llm_samples = 3
        self.ppl_threshold = 150.0 
        self.safety_reward_beta = 0.1
        self.use_safety_prompt = True 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seeds = [42]
        self.refusal_token_id = 50256 

# --- data_utils.py ---
def get_wiki_loaders(config, dataset_name='wikitext-2-raw-v1'):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        from datasets import load_dataset
        hf_path = 'wikitext-103-raw-v1' if '103' in dataset_name else 'wikitext-2-raw-v1'
        dataset = load_dataset('wikitext', hf_path, split='test', trust_remote_code=True)
        dataset = dataset.filter(lambda x: len(x['text']) > 50)
    except (ImportError, Exception):
        # Fallback if datasets is