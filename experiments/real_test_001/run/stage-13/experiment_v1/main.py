```python
filename:config.py
import torch

class ExperimentConfig:
    """Hyperparameter configuration for LLM security experiments."""
    def __init__(self):
        # Training params
        self.lr = 1e-4 # Lowered for RL stability
        self.batch_size = 4 
        self.max_seq_len = 32
        self.num_epochs = 1
        self.train_samples = 12
        self.eval_samples = 8
        
        # Security params
        self.adv_strengths = [0.0, 0.1, 0.2, 0.4] 
        self.smooth_llm_samples = 3