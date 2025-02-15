import torch
import torch.nn as nn

from my_ai.components.block import Block
# Hyperparameters used: n_embd, block_size, n_head

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, hyperparameters={}):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.token_embedding_table = nn.Embedding(vocab_size, hyperparameters['n_embd'])
        self.position_embedding_table = nn.Embedding(hyperparameters['block_size'], hyperparameters['n_embd'])
        self.blocks = nn.Sequential(*[Block(hyperparameters) for _ in range(hyperparameters['n_layer'])])
        self.ln_f = nn.LayerNorm(hyperparameters['n_embd'])
        self.lm_head = nn.Linear(hyperparameters['n_embd'], vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index