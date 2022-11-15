##################################################
# Module for loading and preprocessing datasets
##################################################

import pandas as pd
import numpy as np
import os
import torch
import dataset_utils
import tokenizer_utils


'''
  -  files:  list of review .json files to load
  -  nrows_per_type:  how many reviews to load per .json file (and concatenate)
'''
def load_reviews(files=None, nrows_per_type=None):
    return dataset_utils.load_reviews_df(files, nrows_per_type)

'''
  -  train:  whether to load the training set (True/False)
  -  valid:  whether to load the validation set (True/False)
  -  score_type:  'categorical' (integer score range) or 'binary' (scores: 'True' or 'False') or 'standardized' (in the range [-1.0, 1.0])
'''
def load_essays(train=True, valid=True, test=True, score_type='categorical'):
    dfs = dataset_utils.load_essays_dfs(train, valid, test)
    
    def standardize_df(df, binarize=False):
        def std(df):
            new_df = df.copy()
            new_df['domain1_score'] = 2*((df['domain1_score'] - df['domain1_score'].min()) / (df['domain1_score'].max() - df['domain1_score'].min())) - 1.0
            if binarize:  new_df['domain1_score'] = new_df['domain1_score'] > 0.0
            return new_df
        sub_dfs = []
        for g in df.groupby('essay_set'):
            sub_dfs.append( std(g[1]) )
        return pd.concat(sub_dfs).reset_index()

    if score_type == 'standardized':
        dfs = [standardize_df(df) for df in dfs]
    elif score_type == 'binary':
        dfs = [standardize_df(df, True) for df in dfs]

    return dfs

class ReviewsDataset(torch.utils.data.Dataset):

    '''
      -  tokenizer:  the name of the model file for the tokenizer to load (will train a new one if not found)
      -  force_retrain:  train a new tokenizer and save it, regardless of whether it exists.
      -  BPE_params:  optional parameters for training byte-pair encoder. Check out tokenizer_utils.train_BPE for list of options
    '''
    def __init__(self, tokenizer='reviews_tokenizer', force_retrain=False, BPE_params='default'):
        self.data = load_reviews()
        if force_retrain or tokenizer not in [f.split('.')[0] for f in os.listdir(os.getcwd() + tokenizer_utils.MODEL_DIR)]:
            print("Training BPE tokenizer ...")
            if BPE_params == 'default':  BPE_params = {'lowercase': True, 'vocab_size': 1000}  # default BPE settings
            tokenizer_utils.train_BPE(dataset_utils.REVIEWS_DIR + 'reviews_text.txt', tokenizer, **BPE_params)
        self.tokenizer = tokenizer_utils.load_model(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.tokenizer.encode(self.data['reviewText'][idx]).ids).long(), self.data['overall'][idx]

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

class EssaysDataset(torch.utils.data.Dataset):

    '''
      -  tokenizer:  the name of the model file for the tokenizer to load (will train a new one if not found)
      -  force_retrain:  train a new tokenizer and save it, regardless of whether it exists.
      -  BPE_params:  optional parameters for training byte-pair encoder. Check out tokenizer_utils.train_BPE for list of options
    '''
    def __init__(self, tokenizer='essays_tokenizer', force_retrain=False, BPE_params='default'):
        self.data = load_essays(valid=False, test=False)[0]
        if force_retrain or tokenizer not in [f.split('.')[0] for f in os.listdir(os.getcwd() + tokenizer_utils.MODEL_DIR)]:
            print("Training BPE tokenizer ...")
            if BPE_params=='default':  BPE_params = {'special_tokens': dataset_utils.NER_TOKENS, 'vocab_size': 1000}  # default BPE settings
            tokenizer_utils.train_BPE(dataset_utils.ESSAYS_DIR + 'essays_text.txt', tokenizer, **BPE_params)
        self.tokenizer = tokenizer_utils.load_model(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.data['essay'][idx]).ids, self.data['domain1_score'][idx]

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


# test = EssaysDataset(force_retrain=True)
#ReviewsDataset(force_retrain=True)
d = load_essays(valid=False, test=False, score_type='binary')[0]
print(d[0:20]['domain1_score'])

# print(test[1321][0])