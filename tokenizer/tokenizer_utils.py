'''
tokenizer_utils.py

Module containing functions for training and loading tokenizers.
'''

import os
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

TOKENIZERS_DIR = 'tokenizer'

'''
Train a tokenizer through the Byte-pair encoding (BPE) algorithm
  -  dataset:  the source dataset to train BPE on ('reviews' or 'essays')
  -  model_name:  name of the model (will be in .json)
  -  whitespace:  whether to have the model train on tokens separated by whitespace (True/False)
  -  lowercase:  whether to lowercase the text before training the model (True/False)
  -  special_tokens:  List of strings to add as (permanent) tokens to the model
  -  vocab_size:  The resulting number of tokens the model will generate (and encode with)
'''
def train_BPE(tokenizer_name, dataset, iterator, whitespace=True, lowercase=False, special_tokens=None, vocab_size=None):
    tokenizer = Tokenizer( model=BPE() )
    if(whitespace):  tokenizer.pre_tokenizer = Whitespace()
    if(lowercase):  tokenizer.normalizer = Lowercase()

    if not special_tokens:  special_tokens = []
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)

    tokenizer.save(os.path.join(TOKENIZERS_DIR, dataset, tokenizer_name + '.json'))

def load_tokenizer(tokenizer_name, dataset=None):
    if dataset:
        filepath = os.path.join(TOKENIZERS_DIR, dataset, tokenizer_name + '.json')
    else:
        filepath = os.path.join(TOKENIZERS_DIR, tokenizer_name + '.json')
    return Tokenizer.from_file(filepath)
