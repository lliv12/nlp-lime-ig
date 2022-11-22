##################################################
# Module for loading and preprocessing datasets.
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
  -  score_type:  'categorical' (integer score range) or 'binary' (scores: '0' or '1')
  -  binary_threshold:  (if using binary score)  scores less than or equal to this value are classified as negative (0), and positive (1) otherwise
'''
def load_reviews(files=None, nrows_per_type=None, score_type='categorical', binary_threshold=4):
    df = dataset_utils.load_reviews_df(files, nrows_per_type)

    if score_type == 'standardized':
        raise NotImplementedError("Standardized scores not supported for Reviews data.")
    elif score_type == 'binary':
        df['overall'] = (df['overall'] > binary_threshold).astype('Int64')
    return df

'''
  -  train:  whether to load the training set (True/False)
  -  valid:  whether to load the validation set (True/False)
  -  score_type:  'categorical' (integer score range) or 'binary' (scores: '0' or '1') or 'standardized' (in the range [-1.0, 1.0])
  -  binary_threshold:  (if using binary score)  the threshold above which STANDARDIZED scores are classified as positive (1), and negative (0) otherwise
'''
def load_essays(train=True, valid=True, test=True, score_type='categorical', binary_threshold=0.0):
    dfs = dataset_utils.load_essays_dfs(train, valid, test)

    def categorize_df(df):
        labels = [1, 2, 3, 4]
        slices = [
            pd.cut(df[df['essay_set']==1]['domain1_score'], [1, 7, 8, 9, 12], labels=labels), # Set 1
            pd.cut(df[df['essay_set']==2]['domain1_score'], [0, 2, 3, 4, 6], labels=labels),
            df[df['essay_set']==3]['domain1_score'] + 1,
            df[df['essay_set']==4]['domain1_score'] + 1,
            pd.cut(df[df['essay_set']==5]['domain1_score'], [-1, 1, 2, 3, 4], labels=labels),
            pd.cut(df[df['essay_set']==6]['domain1_score'], [-1, 1, 2, 3, 4], labels=labels),
            pd.qcut(df[df['essay_set']==7]['domain1_score'], 4, labels=labels),
            pd.qcut(df[df['essay_set']==8]['domain1_score'], 4, labels=labels)
        ]
        return df[[c for c in df.columns if c != 'domain1_score']].join(pd.concat(slices))
    
    def standardize_df(df, binarize=False):
        def std(df):
            new_df = df.copy()
            new_df['domain1_score'] = 2*((df['domain1_score'] - df['domain1_score'].min()) / (df['domain1_score'].max() - df['domain1_score'].min())) - 1.0
            if binarize:  new_df['domain1_score'] = (new_df['domain1_score'] > binary_threshold).astype('Int64')
            return new_df
        sub_dfs = []
        for g in df.groupby('essay_set'):
            sub_dfs.append( std(g[1]) )
        return pd.concat(sub_dfs).reset_index()

    if score_type == 'categorical':
        dfs = [categorize_df(df) for df in dfs]
    if score_type == 'standardized':
        dfs = [standardize_df(df) for df in dfs]
    elif score_type == 'binary':
        dfs = [standardize_df(df, True) for df in dfs]

    return dfs


class BaseDataset(torch.utils.data.Dataset):

    class Iterator:
        def __init__(self, base):
            self.idx = 0
            self.base = base
        def __iter__(self):
            return self
        def __next__(self):
            self.idx += 1
            if self.idx <= len(self.base):
                return self.base[self.idx-1]
            raise StopIteration()

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer_utils.load_model(tokenizer)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return BaseDataset.Iterator(self)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

class ReviewsDataset(BaseDataset):

    '''
      -  tokenizer:  the name of the model file for the tokenizer to load (will train a new one if not found)
      -  force_retrain:  train a new tokenizer and save it, regardless of whether it exists.
      -  BPE_params:  optional parameters for training byte-pair encoder. Check out tokenizer_utils.train_BPE for list of options
    '''
    def __init__(self, tokenizer='reviews_tokenizer', force_retrain=False, score_type='categorical', BPE_params='default'):
        self.score_type = score_type
        super().__init__(load_reviews(score_type=score_type), tokenizer)
        if force_retrain or tokenizer not in [f.split('.')[0] for f in os.listdir(os.getcwd() + tokenizer_utils.MODEL_DIR)]:
            print("Training BPE tokenizer ...")
            if BPE_params == 'default':  BPE_params = {'lowercase': True, 'vocab_size': 1000}  # default BPE settings
            tokenizer_utils.train_BPE('reviews', tokenizer, **BPE_params)

    def __getitem__(self, idx, raw=False):
        if raw:
            return self.tokenizer.encode(self.data['reviewText'][idx]), self.data['overall'][idx]
        else:
            if self.score_type in ['categorical', 'binary']:
                sub = 1 if self.score_type == 'categorical' else 0
                label = torch.from_numpy(np.array(self.data['overall'][idx]-sub)).unsqueeze(dim=0)
            else:
                label = torch.from_numpy(np.array(self.data['overall'][idx], dtype='float32'))
            return torch.Tensor(self.tokenizer.encode(self.data['reviewText'][idx]).ids).long(), label

    def get_text(self, idx):
        return self.data['reviewText'][idx]

class EssaysDataset(BaseDataset):

    '''
      -  tokenizer:  the name of the model file for the tokenizer to load (will train a new one if not found)
      -  force_retrain:  train a new tokenizer and save it, regardless of whether it exists.
      -  BPE_params:  optional parameters for training byte-pair encoder. Check out tokenizer_utils.train_BPE for list of options
    '''
    def __init__(self, tokenizer='essays_tokenizer', force_retrain=False, score_type='categorical', BPE_params='default'):
        self.score_type = score_type
        super().__init__(load_essays(valid=False, test=False, score_type=score_type)[0], tokenizer)
        if force_retrain or tokenizer not in [f.split('.')[0] for f in os.listdir(os.getcwd() + tokenizer_utils.MODEL_DIR)]:
            print("Training BPE tokenizer ...")
            if BPE_params=='default':  BPE_params = {'special_tokens': dataset_utils.NER_TOKENS, 'vocab_size': 1000}  # default BPE settings
            tokenizer_utils.train_BPE('essays', tokenizer, **BPE_params)

    def __getitem__(self, idx, raw=False):
        if raw:
            return self.tokenizer.encode(self.data['essay'][idx]), self.data['domain1_score'][idx]
        else:
            if self.score_type in ['categorical', 'binary']:
                sub = 1 if self.score_type == 'categorical' else 0
                label = torch.from_numpy(np.array(self.data['domain1_score'][idx]-sub)).unsqueeze(dim=0)
            else:
                label = torch.from_numpy(np.array(self.data['domain1_score'][idx], dtype='float32'))
            return torch.Tensor(self.tokenizer.encode(self.data['essay'][idx]).ids).long(), label

    def get_text(self, idx):
        return self.data['essay'][idx]