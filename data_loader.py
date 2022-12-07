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
    def __init__(self, data, tokenizer, score_type, seq_len, load_mode, x_column, y_column, cat_column):
        self.data = data
        self.score_type = score_type
        self.tokenizer = tokenizer_utils.load_model(tokenizer)
        self.load_mode = load_mode
        self.x_column = x_column
        self.y_column = y_column
        self.cat_column = cat_column

        # use the max sequence length (in chars) as a heuristic. Get the number of tokens in its encoding.
        self.max_seq_len = len(self.tokenizer.encode( self.__get_x( self.__get_x(range(len(self.data))).str.len().argmax() ) ))
        if seq_len:
            length = (self.max_seq_len if seq_len=='max' else seq_len)
            self.tokenizer.enable_padding(pad_token=dataset_utils.PADDING_TOKEN, length=length)
            self.tokenizer.enable_truncation(length)

        if self.load_mode == 'cache':
            self.prepared_data = [self.__prepare_ex(i) for i in range(len(self.data))]

    '''
      -  idx:  the index of the item in this dataset
      -  format:  the format of the item  ('train': in Tensor format; 'encoding': the tokenizer encoding; 'raw': the raw text)
    '''
    def __getitem__(self, idx, format='train'):
        if format == 'train':
            if self.load_mode == 'cache':
                return self.prepared_data[idx]
            elif self.load_mode == 'lazy':
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
                result = result.loc[result[self.x_column].str.len() >= length]
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

    def __get_x(self, idx):
        return self.data[self.x_column][idx]

    def __get_y(self, idx):
        return self.data[self.y_column][idx]

    def __prepare_ex(self, idx):
        if self.score_type in ['categorical', 'binary']:
            sub = 1 if self.score_type == 'categorical' else 0
            label = torch.from_numpy(np.array(self.__get_y(idx)-sub))
        else:
            label = torch.from_numpy(np.array(self.__get_y(idx), dtype='float32'))
        return torch.Tensor(self.tokenizer.encode(self.__get_x(idx)).ids).long(), label


class ReviewsDataset(BaseDataset):

    '''
      -  BPE_params:  optional parameters for training byte-pair encoder. Check out tokenizer_utils.train_BPE for list of options
    '''
    def __init__(self, tokenizer='reviews_tokenizer', force_retrain=False, score_type='categorical', seq_len=None, load_mode='cache', BPE_params='default'):
        # train a tokenizer if we must or if there's no tokenizer of the given name.
        if force_retrain or tokenizer not in [f.split('.')[0] for f in os.listdir(os.getcwd() + tokenizer_utils.MODEL_DIR)]:
            print("Training BPE tokenizer ...")
            if BPE_params == 'default':  BPE_params = {'lowercase': True, 'vocab_size': 1000}  # default BPE settings
            tokenizer_utils.train_BPE('reviews', tokenizer, **BPE_params)
        super().__init__(load_reviews(score_type=score_type), tokenizer, score_type, seq_len, load_mode, 'reviewText', 'overall', 'type')


class EssaysDataset(BaseDataset):

    '''
      -  BPE_params:  optional parameters for training byte-pair encoder. Check out tokenizer_utils.train_BPE for list of options
    '''
    def __init__(self, tokenizer='essays_tokenizer', force_retrain=False, score_type='categorical', seq_len=None, load_mode='cache', BPE_params='default'):
        # train a tokenizer if we must or if there's no tokenizer of the given name.
        if force_retrain or tokenizer not in [f.split('.')[0] for f in os.listdir(os.getcwd() + tokenizer_utils.MODEL_DIR)]:
            print("Training BPE tokenizer ...")
            if BPE_params=='default':  BPE_params = {'vocab_size': 1000}  # default BPE settings
            tokenizer_utils.train_BPE('essays', tokenizer, **BPE_params)
        super().__init__(load_essays(valid=False, test=False, score_type=score_type)[0], tokenizer, score_type, seq_len, load_mode, 'essay', 'domain1_score', 'essay_set')
