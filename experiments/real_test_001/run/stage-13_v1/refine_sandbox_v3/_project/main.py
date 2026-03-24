# model_zoo.py
from transformers import GPT2LMHeadModel, GPT2Config
# ...
class SecurityModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        # Fast config for experiment
        config = GPT2Config(
            vocab_size=50257,
            n_positions=128,
            n_embd=128,
            n_layer=2,
            n_head=2
        )
        self.model = GPT2LMHeadModel(config)


if __name__ == "__main__":
    run_experiment()