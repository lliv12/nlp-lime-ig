'''
tokenizer.py

Create a new tokenizer for one of the datasets. Use this schema:

python tokenizertokenizer.py <dataset> <tokenizer_name> --tokenizer_type --whitespace --lowercase --vocab_size --essays_sets
  + dataset:  which dataset to tokenize ('reviews' or 'essays')
  + tokenizer_name:  name of the tokenizer (will save <tokenizer_name>.json after training is finished)
  --tokenizer_type:  the type of tokenizer to train  (Ex: 'bpe')
  --whitespace:  whether or not to train the model on tokens separated by whitespace (True/False)
  --lowercase:  whether or not to lowercase the text before training the model (True/False)
  --vocab_size:  the resulting number of tokens the model will generate (and encode with)
  --essays_sets:  which sets to source from from the essays to create the tokenizer
'''

from tokenizer_utils import *
from data.dataset import reviews_source_generator, essays_source_generator, PADDING_TOKEN, NER_TOKENS
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer with a given vocabulary size and name.")
    parser.add_argument('dataset', choices=['reviews', 'essays'], help="The source dataset to train the tokenizer on.")
    parser.add_argument('tokenizer_name', help="Name of the tokenizer  (will save <tokenizer_name>.json after training is finished).")
    parser.add_argument('-t', '--tokenizer_type', default='bpe', choices=['bpe'], help="The type of tokenizer to train.")
    parser.add_argument('-w', '--whitespace', type=bool, default=True, help="Whether or not to train the model on tokens separated by whitespace (True/False).")
    parser.add_argument('-l', '--lowercase', type=bool, default=True, help="Whether or not to lowercase the text before training the model (True/False).")
    parser.add_argument('-v', '--vocab_size', type=int, default=1000, help="The resulting number of tokens the model will generate (and encode with).")
    parser.add_argument('--essays_sets', nargs='+', choices=['train', 'val', 'test'], help="which sets to source from from the essays to create the tokenizer")

    args = parser.parse_args()

    special_tokens = [PADDING_TOKEN]
    if args.dataset == 'essays':
        iterator = essays_source_generator(**{key: True for key in args.essays_sets})
        special_tokens += NER_TOKENS
    else:
        iterator = reviews_source_generator()
    if args.tokenizer_type == 'bpe':
        train_BPE(args.tokenizer_name, iterator, args.whitespace, args.lowercase, special_tokens, args.vocab_size)
    else:
        raise NotImplementedError(f"Unrecognized tokenizer type: {args.tokenizer_type}")