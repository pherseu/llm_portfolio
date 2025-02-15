import torch
import torch.nn as nn

from my_ai.components.head import Head

#hyper parameters used: n_embd, dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, hyperparameters={}):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, hyperparameters=hyperparameters) for _ in range(hyperparameters['n_head'])])
        self.proj = nn.Linear(head_size * hyperparameters['n_head'], hyperparameters['n_embd'])
        self.dropout = nn.Dropout(hyperparameters['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out