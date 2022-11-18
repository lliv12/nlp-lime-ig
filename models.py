from torch import nn
import torch
import os

MODEL_DIR = "/saved_models/"


def save_model(model, name):
    torch.save(model, os.getcwd() + MODEL_DIR + name + '.pt')

def load_model(model_name):
    return torch.load(os.getcwd() + MODEL_DIR + model_name + '.pt')

class BasicDANModel(nn.Module):

    def __init__(self, vocab_size, emb_dim=32, out_size=1, bin=False):
        super().__init__()
        self.bin = bin
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(emb_dim, out_size)
        if bin:  self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_( self.out.weight )

    # For IG interpolation
    def get_embeddings(self, input):
        return self.emb(input).unsqueeze(dim=0)

    # NOTE: input must be (num_tokens, emb_dim)
    def forward(self, input):
        out = self.out( self.tanh(torch.mean(self.emb(input), dim=0)) )
        return self.sigmoid(out) if self.bin else out

    # NOTE: emb must be (num_examples, num_tokens, emb_dim)
    def forward_emb(self, emb):
        out = self.out( self.tanh(torch.mean(emb, dim=1)) )
        return self.sigmoid(out) if self.bin else out