##################################################
# NN Model definitions.
##################################################

from torch import nn
import torch
import os
import numpy as np

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

    # [For IG interpolation]  NOTE: input must be (batch_size, num_tokens)
    def get_embeddings(self, input):
        return self.emb(input)

    # input:  (batch_size, num_tokens)
    def forward(self, input):
        return self.out( self.tanh(torch.mean(self.emb(input), dim=1)) )

    # NOTE: emb must be (num_examples, num_tokens, emb_dim)
    def forward_emb(self, emb):
        return self.out( self.tanh(torch.mean(emb, dim=1)) )

class TransformerModel(nn.Module):

    ''' 
      -  vocab_size:  # tokens from tokenization
      -  emb_dim:  dimension of token embeddings
      -  nhead:  the number of heads in the transformer layer
      -  num_encoder_layers:  the number of transformer encoder layers to stack
      -  dim_feedforward:  the dimension of the feedforward output for each intermediate transformer encoder layer
      -  out_size:  the output dimension of the whole network
    '''
    def __init__(self, vocab_size, emb_dim=32, nhead=1, num_encoder_layers=1, dim_feedforward=512, out_size=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.Linear(emb_dim, out_size)
        self.init_weights()

    # [For IG interpolation]  NOTE: input must be (num_tokens, emb_dim)
    def get_embeddings(self, input):
        return self.embedding(input)

    # input:  (batch_size, num_tokens)
    def forward(self, input):
        src_emb = self.embedding(input) * np.sqrt(self.vocab_size)
        padding_mask = ~input.bool()#(~input.bool()).long()
        return self.decoder( self.encoder(src_emb, src_key_padding_mask=padding_mask) )[:,-1]

    def forward_emb(self, emb):
        emb = emb * np.sqrt(self.vocab_size)
        return self.decoder( self.encoder(emb) ).squeeze(dim=0)[-1]

    def init_weights(self, range=0.1):
        # This initialization scheme was found here:  https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        self.embedding.weight.data.uniform_(-range, range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-range, range)
