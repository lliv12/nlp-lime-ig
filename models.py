from torch import nn
import torch

class BasicDANModel(nn.Module):

    def __init__(self, vocab_size, emb_dim=32, out_size=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(emb_dim, out_size)
        nn.init.xavier_uniform_( self.out.weight )

    def forward(self, input):
        return self.out( self.tanh(torch.mean(self.emb(input), dim=0)) )