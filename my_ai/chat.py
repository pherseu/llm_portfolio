import pickle
import torch
from my_ai.core import AiCore

class Chatbot(AiCore):
    def __init__(self, hyperparameters={}):
        super().__init__(hyperparameters=hyperparameters)
        
        self.model = self.start_model()
        print('loading model parameters...')

        with open('model-01.pkl', 'rb') as f:
            self.model = pickle.load(f)
        print('loaded successfully')

        m = self.model.to(self.device)

        while True:
            prompt = input('Prompt:\n')
            context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)
            generated_chars = self.decode(m.generate(context.unsqueeze(0), max_new_tokens=500, block_size=hyperparameters['block_size'])[0].tolist())
            print(f'Completion:\n{generated_chars}')