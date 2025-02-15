import torch
import torch.nn as nn

# Hyperparamenters used: n_embd, block_size, dropout

class Head(nn.Module):
    def __init__(self, head_size, hyperparameters={}):
        super().__init__()
        self.key = nn.Linear(hyperparameters['n_embd'], head_size, bias=False)
        self.query = nn.Linear(hyperparameters['n_embd'], head_size, bias=False)
        self.value = nn.Linear(hyperparameters['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(hyperparameters['block_size'], hyperparameters['block_size'])))

        self.dropout = nn.Dropout(hyperparameters['dropout'])

    def forward(self, x):
        B, T, C = x.shape

        # if self.tril.shape[0] < T:
        #     self.tril = torch.tril(torch.ones(T, T)).to(x.device) 

        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out