'''
models.py

Model definitions and functions for loading / saving models.
'''

from torch import nn
import torch
import numpy as np
import math
import os

MODEL_DIR = "saved_models"
REVIEWS_SUBDIR = 'reviews'
ESSAYS_SUBDIR = 'essays'


def save_model(model, model_name, tokenizer_name):
    model_dict = {'model': model, 'tokenizer': tokenizer_name}
    torch.save(model_dict, os.path.join(MODEL_DIR, model_name + '.pt'))

'''
(if model_type):  Returns:  model
(if model_name):  Returns:  {model, tokenizer}
'''
def load_model(model_type=None, model_name=None, **kwargs):
    if model_type:
        if model_type == 'BasicDANModel':
            return BasicDANModel(**kwargs)
        elif model_type == 'TransformerModel':
            return TransformerModel(**kwargs)
        else:
            raise Exception("Unknown model type: '{m}'".format(m=model_type))
    elif model_name:
        load_dir = os.path.join(MODEL_DIR, model_name + '.pt')
        if torch.cuda.is_available():
            return torch.load(load_dir)
        else:
            return torch.load(load_dir, map_location=torch.device('cpu'))
    else:
        raise Exception("Must specify 'model_type' or 'model_name'.")

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
    def __init__(self, vocab_size, emb_dim=128, nhead=8, num_encoder_layers=6, dim_feedforward=512, out_size=1):
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
        padding_mask = ~input.bool()
        return self.decoder( self.encoder(src_emb, src_key_padding_mask=padding_mask) )[:,-1]

    def forward_emb(self, emb):
        emb = emb * np.sqrt(self.vocab_size)
        return self.decoder( self.encoder(emb) ).squeeze(dim=0)[-1]

    def init_weights(self, range=0.1):
        # This initialization scheme was found here:  https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        self.embedding.weight.data.uniform_(-range, range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-range, range)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
