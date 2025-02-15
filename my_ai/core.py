import torch
import torch.nn as nn

from my_ai.components.gpt import GPTLanguageModel

class AiCore():
    def __init__(self, hyperparameters={}):
        self.hyperparameters = hyperparameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.chars = ''
        with open('vocab.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            chars = sorted(list(set(text)))

        self.vocab_size = len(chars)

        # encoding and decoding
        self.string_to_int = { ch:i for i, ch in enumerate(chars) }
        self.int_to_string = { i:ch for i, ch in enumerate(chars) }
        self.encode = lambda s: [self.string_to_int[c] for c in s]
        self.decode = lambda l: ''.join([self.int_to_string[i] for i in l])

        
    def start_model(self):
        return GPTLanguageModel(vocab_size = self.vocab_size, hyperparameters=self.hyperparameters)