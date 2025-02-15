import torch.nn as nn

from my_ai.components.multi_head import MultiHeadAttention
from my_ai.components.feed_forward import FeedForward

class Block(nn.Module):
    def __init__(self, hyperparameters={}):
        super().__init__()
        head_size = hyperparameters['n_embd'] // hyperparameters['n_head']
        self.sa = MultiHeadAttention(hyperparameters=hyperparameters, head_size=head_size)
        self.ffwd = FeedForward(hyperparameters=hyperparameters)
        self.ln1 = nn.LayerNorm(hyperparameters['n_embd'])
        self.ln2 = nn.LayerNorm(hyperparameters['n_embd'])

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x