from torch import nn
import torch
import os

MODEL_DIR = "/saved_models/"


def save_model(model, name):
    torch.save(model, os.getcwd() + MODEL_DIR + name + '.pt')

def load_model(model_name):
    return torch.load(os.getcwd() + MODEL_DIR + model_name + '.pt')

class BasicDANModel(nn.Module):

    def __init__(self, vocab_size, emb_dim=32, out_size=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(emb_dim, out_size)
        nn.init.xavier_uniform_( self.out.weight )

    def forward(self, input):
        return self.out( self.tanh(torch.mean(self.emb(input), dim=0)) )