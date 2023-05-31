'''
dataset.py

Dataset modules and utilities are located in here.
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tokenizer.tokenizer_utils import TOKENIZERS_DIR, load_tokenizer, train_BPE
import torch
import numpy as np
import re


AMAZON_DATASET_DIR = "dataset/amazon"
KAGGLE_DATASET_DIR = "dataset/kaggle"

KAGGLE_DATASET_FILES = {
    'train': 'training_set_rel3.xls',
    'val': 'valid_set.xls',
    'test': 'test_set.tsv'
}

PADDING_TOKEN = '<PAD>'
NER_TOKENS = ['@CAPS', '@CITY', '@DATE', '@DR', '@EMAIL', '@LOCATION', '@MONEY', '@MONTH', '@NUM', '@ORGANIZATION', '@PERCENT', '@PERSON', '@STATE', '@TIME']

def load_reviews(files=None):
    if not files:
        files = [f for f in os.listdir(AMAZON_DATASET_DIR) if f.split('.')[1]=='json']
    dfs = []
    for f in files:
        df = pd.read_json(os.path.join(AMAZON_DATASET_DIR, f))
        df['type'] = f.split('.')[0]
        dfs.append(df)
    return pd.concat(dfs).dropna(subset=['reviewText', 'overall']).reset_index()

def load_essays(train=True, valid=False, test=False):
    results = []
    if train:
        results.append( pd.read_excel(os.path.join(KAGGLE_DATASET_DIR, KAGGLE_DATASET_FILES['train'])) )
    if valid:
        results.append( pd.read_excel(os.path.join(KAGGLE_DATASET_DIR, KAGGLE_DATASET_FILES['val'])) )
    if test:
        results.append( pd.read_csv(os.path.join(KAGGLE_DATASET_DIR, KAGGLE_DATASET_FILES['test']), delimiter='\t', encoding='ISO-8859-1') )

    results = [preprocess_essays(df) for df in results]

    return results if len(results) > 1 else results[0]

def preprocess_essays(df):
    # Pre-process NER tokens (separate numeral from the capitalized part)
    df = df.replace(['', '<NA>'], np.nan)
    for col in df.columns:
        if col != 'essay':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    df = df.dropna(subset=['essay', 'domain1_score']).reset_index()
    df['essay'] = df['essay'].transform(func=lambda s: re.sub('|'.join(NER_TOKENS), lambda x: x.group() + ' ', s) if type(s) == str else s)
    
    # discretize essay scores into 4 bins
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

def reviews_source_generator():
    data = load_reviews()
    for _, row in data.iterrows():
        yield row['reviewText']

def essays_source_generator():
    data = load_essays(train=True)
    for _, row in data.iterrows():
        yield row['essay']


class BaseDataset(Dataset):
    '''
      -  tokenizer:  the name of the model file for the tokenizer to load (will train a new one if not found)
      -  force_retrain:  train a new tokenizer and save it, regardless of whether it exists.
      -  score_type:  the type of score to use
      -  seq_len:  the set length for sequences (in #tokens). Will cut down larger sequences, and pad shorter sequences. If not set, will leave sequences alone.
                   If 'max', will pad sequences to the max sequence length.
      -  load_mode:  the mode for loading the data ('cache': tokenize examples internally; best used for training. 'lazy': )
      -  get_x_func:  function for retrieving text input at specific index (from pandas DataFrame)
      -  get_y_func:  function for retrieving label at specific index (from pandas DataFrame)
    '''
    def __init__(self, data, tokenizer, x_column, y_column, cat_column):
        self.data = data
        self.tokenizer = load_tokenizer(tokenizer)
        self.x_column = x_column
        self.y_column = y_column
        self.cat_column = cat_column

    '''
      -  idx:  the index of the item in this dataset
      -  format:  the format of the item  ('train': in Tensor format; 'encoding': the tokenizer encoding; 'raw': the raw text)
    '''
    def __getitem__(self, idx, format='train'):
        if format == 'train':
            return self.__prepare_ex(idx)
        elif format == 'encoding':
            return self.tokenizer.encode(self.__get_x(idx)), self.__get_y(idx)
        elif format == 'raw':
            return self.__get_x(idx), self.__get_y(idx)
        else:
            raise NotImplementedError("Unsupported format:  '{f}'".format(f=format))

    def __len__(self):
        return len(self.data)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_unique_labels(self):
        return self.data[self.y_column].unique()

    def get_unique_categories(self):
        return self.data[self.cat_column].unique()

    # Load random examples based on the filter values
    # category:  filter by category (for reviews: 'type';  for essays: 'essay_set') 
    # length:    ('int' or 'List[int]')  max length of the example; or bounds of lengths inclusive (in chars)
    # score:     ('int' or 'List[int]' or 'List[float]')  score of the example or bounds of scores inclusive (must use bounds if score_type == 'standardized')
    # format:    (default: 'train')  The format of the example you want to return
    def filter_load(self, category=None, length=None, score=None, n_samples=1, format='train'):
        result = self.data
        if category:
            result = result.loc[result[self.cat_column] == category]
        if length:
            if type(length) == int:
                result = result.loc[result[self.x_column].str.len() <= length]
            elif type(length) == list:
                result = result.loc[(result[self.x_column].str.len() >= length[0]) &
                                    (result[self.x_column].str.len() <= length[1])]
            else: raise Exception("The datatype of 'length' must be 'int' or 'List';  not '{t}'.".format(t=type(length)))
        if score != None:
            if type(score) == int:
                if self.score_type == 'standardized':  raise Exception("Cannot filter by 'int' score for this dataset. Use List['float'] instead.")
                result = result.loc[result[self.y_column] == score]
            elif type(score) == list:
                result = result.loc[(result[self.y_column] >= score[0]) &
                                    (result[self.y_column] <= score[1])]
            else: raise Exception("The datatype of 'score' must be 'int' or 'List';  not '{t}'.".format(t=type(length)))
        sample = result.sample(n_samples).index
        if format == 'index':
            return list(sample)
        return [self.__getitem__(idx, format) for idx in sample]

    def encode(self, text, mode='encode'):
        if type(text) == str:
            enc = self.tokenizer.encode(text)
        else:
            enc = self.tokenizer.encode_batch(text)
        if mode == 'train':
            enc = torch.Tensor([e.ids for e in enc]).long()
        return enc

    def compute_class_weights(self):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)
        vcounts_inv = 1 - self.data[self.y_column].value_counts(normalize=True)
        return torch.tensor(softmax(vcounts_inv.sort_index()).values, dtype=torch.float)

    def __get_x(self, idx):
        return self.data[self.x_column][idx]

    def __get_y(self, idx):
        return self.data[self.y_column][idx]

    def __prepare_ex(self, idx):
        label = torch.from_numpy(np.array(self.__get_y(idx)-1))
        return torch.Tensor(self.tokenizer.encode(self.__get_x(idx)).ids).long(), label


class ReviewsDataset(BaseDataset):
    '''
      -  BPE_params:  optional parameters for training byte-pair encoder. Check out tokenizer_utils.train_BPE for list of options
    '''
    def __init__(self, tokenizer='reviews_tokenizer', force_retrain=False):
        # train a tokenizer if we must or if there's no tokenizer of the given name.
        if force_retrain or tokenizer not in [f.split('.')[0] for f in os.listdir(TOKENIZERS_DIR)]:
            print("Training BPE tokenizer ...")
            special_tokens = [PADDING_TOKEN]
            BPE_params = {'lowercase': True, 'vocab_size': 1000, 'special_tokens': special_tokens}  # default BPE settings
            train_BPE(tokenizer, reviews_source_generator(), **BPE_params)
        super().__init__(load_reviews(), tokenizer, 'reviewText', 'overall', 'type')


class EssaysDataset(BaseDataset):
    '''
      -  BPE_params:  optional parameters for training byte-pair encoder. Check out tokenizer_utils.train_BPE for list of options
    '''
    def __init__(self, tokenizer='essays_tokenizer', force_retrain=False):
        # train a tokenizer if we must or if there's no tokenizer of the given name.
        if force_retrain or tokenizer not in [f.split('.')[0] for f in os.listdir(TOKENIZERS_DIR)]:
            print("Training BPE tokenizer ...")
            special_tokens = [PADDING_TOKEN] + NER_TOKENS
            BPE_params = {'vocab_size': 1000, 'special_tokens': special_tokens}  # default BPE settings
            train_BPE(tokenizer, essays_source_generator(), **BPE_params)
        super().__init__(load_essays(), tokenizer, 'essay', 'domain1_score', 'essay_set')


class BucketSampler(Sampler):

    def __init__(self, data_source: Dataset, batch_size: int, shuffle: bool = True):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            np.random.shuffle(indices)

        # sort indices by sequence length
        indices.sort(key=lambda i: len(self.data_source[i][0]))

        # create batches of indices
        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]

        # shuffle the batches
        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

def padding_collate_func(data):
    # data is a list of tuples (sequence, label)
    sequences = [item[0] for item in data]
    labels = torch.stack([item[1] for item in data])
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    return sequences_padded, labels
