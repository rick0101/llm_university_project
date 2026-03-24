import torch

class ExperimentConfig:
    """Hyperparameter configuration for LLM security experiments."""
    def __init__(self):
        # Training params
        self.lr = 3e-5
        self.batch_size = 4 
        self.max_seq_len = 32
        self.num_epochs = 1
        self.train_samples = 16
        self.eval_samples = 12
        
        # Security params
        self.adv_strengths = [0.0, 0.1, 0.2, 0.4] 
        self.smooth_llm_samples = 3
        self.ppl_threshold = 150.0 
        self.safety_reward_beta = 0.1
        self.use_safety_prompt = True 
        
        # Environment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seeds = [42]
        self.refusal_token_id = 50256 # GPT-2 EOS/Pad as refusal

HYPERPARAMETERS = {
    'learning_rate': 3e-5,
    'batch_size': 4,
    'smooth_samples': 3,
    'safety_beta': 0.1
}