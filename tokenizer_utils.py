##########################################################
# Utilities for training and loading tokenizer models
##########################################################

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from models import MODEL_DIR
from dataset_utils import REVIEWS_DIR, ESSAYS_DIR, PADDING_TOKEN, NER_TOKENS
import argparse
import os

REVIEWS_SRC = REVIEWS_DIR + 'reviews_text.txt'
ESSAYS_SRC = ESSAYS_DIR + 'essays_text.txt'


'''
  -  dataset:  the source dataset to train BPE on ('reviews' or 'essays')
  -  model_name:  name of the model (will be in .json)
  -  whitespace:  whether to have the model train on tokens separated by whitespace (True/False)
  -  lowercase:  whether to lowercase the text before training the model (True/False)
  -  special_tokens:  List of strings to add as (permanent) tokens to the model
  -  vocab_size:  The resulting number of tokens the model will generate (and encode with)
'''
def train_BPE(dataset, model_name, whitespace=True, lowercase=False, special_tokens=None, vocab_size=None):
    tokenizer = Tokenizer( model=BPE() )
    if(whitespace):  tokenizer.pre_tokenizer = Whitespace()
    if(lowercase):  tokenizer.normalizer = Lowercase()

    if not special_tokens:  special_tokens = []
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    tokenizer.train(files=[REVIEWS_SRC if dataset == 'reviews' else ESSAYS_SRC], trainer=trainer)

    if model_name not in [f.split('.')[0] for f in os.listdir(os.getcwd() + MODEL_DIR)]:
        with open(os.getcwd() + MODEL_DIR + model_name + '.json', 'w') as file:  pass
    tokenizer.save(os.getcwd() + MODEL_DIR + model_name + '.json')

def load_model(model_name):
    return Tokenizer.from_file(os.getcwd() + MODEL_DIR + model_name + '.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer with a given vocabulary size and name.")
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="The source dataset to train the tokenizer on.")
    parser.add_argument('-n', '--tokenizer_name', help="Name of the tokenizer  (will save <tokenizer_name>.json after training is finished).")
    parser.add_argument('-t', '--tokenizer_type', default='bpe', choices=['bpe'], help="The type of tokenizer to train.")
    parser.add_argument('-w', '--whitespace', type=bool, default=True, help="Whether or not to train the model on tokens separated by whitespace (True/False).")
    parser.add_argument('-l', '--lowercase', type=bool, default=True, help="Whether or not to lowercase the text before training the model (True/False).")
    parser.add_argument('-v', '--vocab_size', type=int, default=1000, help="The resulting number of tokens the model will generate (and encode with).")

    args = parser.parse_args()

    special_tokens = [PADDING_TOKEN]
    if args.dataset == 'essays':  special_tokens += NER_TOKENS
    if args.tokenizer_type == 'bpe':
        train_BPE(args.dataset, args.tokenizer_name, args.whitespace, args.lowercase, special_tokens, args.vocab_size)