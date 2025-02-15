import torch.nn as nn

# Hyperparamenters used: dropout, n_embd

class FeedForward(nn.Module):
    def __init__(self, hyperparameters={}):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hyperparameters['n_embd'], 4 * hyperparameters['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * hyperparameters['n_embd'], hyperparameters['n_embd']),
            nn.Dropout(hyperparameters['dropout']),
        )

    def forward(self, x):
        return self.net(x)