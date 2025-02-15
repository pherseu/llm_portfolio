import mmap
import pickle
import random
import torch

from my_ai.core import AiCore

class Training(AiCore):
    def __init__(self, hyperparameters={}):
        super().__init__(hyperparameters=hyperparameters)

        self.hp = hyperparameters

        self.model = self.start_model()
        self.m = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hp['learning_rate'])

        for iter in range(self.hp['max_iters']):
            if iter % self.hp['eval_interval'] == 0:
                self.losses = self.estimate_loss()
                print(f'steps: {iter}, train loss; {self.losses['train']:.3f}, val loss: {self.losses['val']:3f}')
            
            self.xb, self.yb = self.get_batch('train')

            self.logits, self.loss = self.model.forward(self.xb, self.yb)
            self.optimizer.zero_grad(set_to_none=True)
            self.loss.backward()
            self.optimizer.step()
        print(self.loss.item())

        with open('model-01.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print('model saved')

# # training
    def get_random_chunk(self, split):
        filename = './_dataset/train_split.txt' if split == 'train' else './_dataset/val_split.txt'
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                file_size = len(mm)
                start_pos = random.randint(0, (file_size) - self.hp['block_size'] * self.hp['batch_size'])

                mm.seek(start_pos)
                block = mm.read(self.hp['block_size'] * self.hp['batch_size']-1)

                decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

                data = torch.tensor(self.encode(decoded_block), dtype=torch.long)
        return data

    def get_batch(self, split):
        data = self.get_random_chunk(split)
        ix = torch.randint(len(data) - self.hp['block_size'], (self.hp['batch_size'],))
        x = torch.stack([data[i:i+self.hp['block_size']] for i in ix])
        y = torch.stack([data[i+1: i+self.hp['block_size']+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    # loss calculation
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.hp['eval_interval'])
            for k in range(self.hp['eval_interval']):
                X, Y = self.get_batch(split)
                self.logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = loss.item()
        self.model.train()
        return out