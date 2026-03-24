import torch

class ExperimentConfig:
    """Hyperparameter configuration for LLM security experiments."""
    def __init__(self):
        # Training params
        self.lr = 1e-5
        self.batch_size = 4  # Small batch for speed and memory
        self.max_seq_len = 32
        self.num_epochs = 1
        self.train_samples = 20  # Fast fine-tuning for demo
        self.eval_samples = 20
        
        # Security params
        self.adversarial_epsilon = 0.05
        self.smooth_llm_samples = 4
        self.ppl_threshold = 4.5  # Log-scale perplexity threshold
        self.safety_reward_beta = 0.05
        
        # Environment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seeds = [42, 123, 456] # Minimum 3 seeds
        self.budget_seconds = 480 # 8 minutes guard
        
        # Refusal token (using GPT-2's "I" token as a proxy for refusal start)
        self.refusal_token_id = 40 

HYPERPARAMETERS = {
    'learning_rate': 1e-5,
    'batch_size': 4,
    'max_seq_len': 32,
    'adv_epsilon': 0.05,
    'smooth_samples': 4,
    'ppl_threshold': 4.5,
    'safety_beta': 0.05,
    'num_epochs': 1
}