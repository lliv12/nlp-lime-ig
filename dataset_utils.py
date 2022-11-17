##########################################################
# Utilities Module; mainly used for managing data files
##########################################################

import json
import os
import re
import pandas as pd
import numpy as np


MAX_ENTRIES_PER_CATEGORY = 3000
REVIEWS_DIR = "datasets/reviews/"
ESSAYS_DIR = "datasets/asap-aes/"

REVIEWS_FILES = [
    "appliances-5core.json",
    "arts_crafts_and_sewing-5core.json",
    "automotive-5core.json",
    "books-5core.json",
    "cellphones_and_accessories-5core.json",
    "clothing_shoes_and_jewelry-5core.json",
    "fashion-5core.json",
    "home_and_kitchen-5core.json",
    "movies_and_tv-5core.json",
    "office_products-5core.json",
    "pet_supplies-5core.json",
    "sports_and_outdoors-5core.json",
    "tools_and_home_improvement-5core.json",
    "video_games-5core.json"
]

NER_TOKENS = ['@CAPS', '@CITY', '@DATE', '@DR', '@EMAIL', '@LOCATION', '@MONEY', '@MONTH', '@NUM', '@ORGANIZATION', '@PERCENT', '@PERSON', '@STATE', '@TIME']


def load_reviews_df(files=None, nrows_per_type=None):
    if not files:
        files = [f for f in os.listdir(REVIEWS_DIR) if f.split('.')[1]=='json']
    dfs = []
    for f in files:
        df = pd.read_json(REVIEWS_DIR + f, lines=True, nrows=nrows_per_type)
        df['type'] = f.split('-')[0]
        dfs.append(df)
    return pd.concat(dfs).reset_index()

def load_essays_dfs(train=True, valid=True, test=True):
    files = []
    if train: files.append('train_set.json')
    if valid: files.append('valid_set.json')
    if test:  files.append('test_set.json')
    # Pre-process NER tokens (separate numeral from the capitalized part)
    def preprocess(df):
        df.replace('', np.nan, inplace=True)
        for col in df.columns:
            if col != 'essay':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        df.dropna(subset=['essay', 'domain1_score'], inplace=True)
        df.reset_index(inplace=True)
        df['essay'] = df['essay'].transform(func=lambda s: re.sub('|'.join(NER_TOKENS), lambda x: x.group() + ' ', s) if type(s) == str else s)
        return df
    dfs = [preprocess(pd.read_json(ESSAYS_DIR + f, lines=False)) for f in files]
    return dfs


def create_reduced_reviews_dataset(out_dir=REVIEWS_DIR):
    # Create shorter dataset using the first <max_entries> reviews
    def dump_json(file_name, max_entries):
        data = []
        cnt = 0
        for line in open('datasets/reviews_full/' + file_name, 'r'):
            data.append( json.loads(line) )
            if cnt == max_entries:  break
            cnt += 1
        with open(os.getcwd() + out_dir + file_name, 'w') as file:
            file.write( json.dumps(data)[1:-1] )

    for f in REVIEWS_FILES:
        dump_json(f, MAX_ENTRIES_PER_CATEGORY)

# Generate a large text file with concatentation of all reviews for training BPE
def create_reviews_text_file(out_dir=REVIEWS_DIR, file_name="reviews_text.txt"):
    df = load_reviews_df()
    with open(os.getcwd() + out_dir + file_name, 'w') as file:
        file.write( ' '.join([df['reviewText'][i] for i in range(len(df)) if type(df['reviewText'][i]) == str]) )

# Generate a large text file with concatentation of all essays for training BPE (training set only)
def create_essays_text_file(out_dir=ESSAYS_DIR, file_name="essays_text.txt"):
    df = load_essays_dfs(valid=False, test=False)
    with open(os.getcwd() + out_dir + file_name, 'w', encoding="utf-8") as file:
        file.write( ' '.join([df['essay'][i] for i in range(len(df)) if type(df['essay'][i]) == str]) )

if __name__ == "__main__":
    # create_reduced_reviews_dataset()
    # create_reviews_text_file()
    # create_essays_text_file()
    load_essays_dfs()