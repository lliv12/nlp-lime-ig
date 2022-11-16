##########################################################
# Utilities for training and loading tokenizer models
##########################################################

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from models import MODEL_DIR
import os


'''
  -  text_file:  the name of the .txt file containing the text you want to train BPE on
  -  model_name:  name of the model (will be in .json)
  -  whitespace:  whether to have the model train on tokens separated by whitespace (True/False)
  -  lowercase:  whether to lowercase the text before training the model (True/False)
  -  special_tokens:  List of strings to add as (permanent) tokens to the model
  -  vocab_size:  The resulting number of tokens the model will generate (and encode with)
'''
def train_BPE(text_file, model_name, whitespace=True, lowercase=False, special_tokens=None, vocab_size=None):
    tokenizer = Tokenizer( model=BPE() )
    if(whitespace):  tokenizer.pre_tokenizer = Whitespace()
    if(lowercase):  tokenizer.normalizer = Lowercase()

    if not special_tokens:  special_tokens = []
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    tokenizer.train(files=[text_file], trainer=trainer)

    if model_name not in [f.split('.')[0] for f in os.listdir(os.getcwd() + MODEL_DIR)]:
        with open(os.getcwd() + MODEL_DIR + model_name + '.json', 'w') as file:  pass
    tokenizer.save(os.getcwd() + MODEL_DIR + model_name + '.json')

def load_model(model_name):
    return Tokenizer.from_file(os.getcwd() + MODEL_DIR + model_name + '.json')