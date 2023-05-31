'''
download_dataset.py

Module to download the Amazon reviews dataset. You can choose which categories will be downloaded, or specify a csv file
containing the reviews categories to be downloaded. Refer to the dictionary below (AMAZON_DICT) for valid review keys.

NOTE: The essays must be downloaded separately from Kaggle. A Kaggle account is required. Please download the zip file
from here:  https://www.kaggle.com/competitions/asap-aes/data and extract into dataset/kaggle.

Schema:
python data/download_dataset.py --cat_spec --cat --limit --verbose
  --cat_spec:  filepath to a csv file containing the review categories to be downloaded. (if --cat is specified, that will have priority)
  --cat:  a list of reviews categories to be downloaded.  (Ex: <review1> <review2> ... <reviewN>)
  --limit:  limit the number of reviews per json file
  --verbose:  whether or not to log progress of downloading to the console (default: True)

(Example)
python data/download_dataset.py --cat digital_music luxury_beauty musical_instruments --limit 5000
'''

import os
import argparse
import requests
import json
import gzip
import shutil
import warnings
import csv
from urllib3.exceptions import InsecureRequestWarning
from data.dataset import AMAZON_DATASET_DIR
import numpy as np

AMAZON_PARENT = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall"
AMAZON_DICT = {
    'amazon_fashion': 'AMAZON_FASHION_5.json.gz',
    'all_beauty': 'All_Beauty_5.json.gz',
    'appliances': 'Appliances_5.json.gz',
    'arts_crafts_and_sewing': 'Arts_Crafts_and_Sewing_5.json.gz',
    'automotive': 'Automotive_5.json.gz',
    'books': 'Books_5.json.gz',
    'cds_and_vinyl': 'CDs_and_Vinyl_5.json.gz',
    'cellphones_and_accessories': 'Cell_Phones_and_Accessories_5.json.gz',
    'clothing_shoes_and_jewelry': 'Clothing_Shoes_and_Jewelry_5.json.gz',
    'digital_music': 'Digital_Music_5.json.gz',
    'electronics': 'Electronics_5.json.gz',
    'gift_cards': 'Gift_Cards_5.json.gz',
    'grocery_and_gourmet_food': 'Grocery_and_Gourmet_Food_5.json.gz',
    'home_and_kitchen': 'Home_and_Kitchen_5.json.gz',
    'industrial_and_scientific': 'Industrial_and_Scientific_5.json.gz',
    'kindle_store': 'Kindle_Store_5.json.gz',
    'luxury_beauty': 'Luxury_Beauty_5.json.gz',
    'magazine_subscriptions': 'Magazine_Subscriptions_5.json.gz',
    'movies_and_tv': 'Movies_and_TV_5.json.gz',
    'musical_instruments': 'Musical_Instruments_5.json.gz',
    'office_products': 'Office_Products_5.json.gz',
    'patio_lawn_and_garden': 'Patio_Lawn_and_Garden_5.json.gz',
    'pet_supplies': 'Pet_Supplies_5.json.gz',
    'prime_pantry': 'Prime_Pantry_5.json.gz',
    'software': 'Software_5.json.gz',
    'sports_and_outdoors': 'Sports_and_Outdoors_5.json.gz',
    'tools_and_home_improvement': 'Tools_and_Home_Improvement_5.json.gz',
    'toys_and_games': 'Toys_and_Games_5.json.gz',
    'video_games': 'Video_Games_5.json.gz',
}


def download_amazon_reviews(categories, limit=None, verbose=True):
    def fix_json(json_file, limit=None):
        data = []
        with open(json_file, 'r') as f:
            cnt = 0
            for line in f:
                if limit and cnt >= limit:
                    break
                data.append(json.loads(line))
                cnt += 1
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)

    if not os.path.exists(AMAZON_DATASET_DIR):
        os.makedirs(AMAZON_DATASET_DIR)

    if verbose:
        print("\nCategories: ", categories)
        if limit:  print(f"Limiting to {limit}")
    for cat in categories:
        endpoint = AMAZON_DICT[cat]
        url = AMAZON_PARENT + '/' + endpoint
        warnings.simplefilter('ignore', InsecureRequestWarning)
        filesize = np.around(int(requests.head(url, verify=False).headers.get('Content-Length', 0)) / (1024**2), 2)
        if verbose: print(f"downloading {endpoint} ({filesize}MB) ...")
        response = requests.get(url, stream=True, verify=False)
        tar_file = os.path.join(AMAZON_DATASET_DIR, endpoint)
        json_file = os.path.join(AMAZON_DATASET_DIR, cat + '.json')

        # write contents of response to tar file (.gz)
        with open(tar_file, 'wb') as f:
            f.write(response.raw.read())
        # write contents to json file
        with gzip.open(tar_file, 'rb') as f_in:
            with open(json_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            fix_json(json_file, limit)
        # remove tar file
        os.remove(tar_file)
    if verbose:  print("\nDone\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat_spec', default='data/amazon_default.csv', help='filepath to a csv file containing the review categories to be downloaded. (if --cat is specified, that will have priority)')
    parser.add_argument('--cat', nargs='+', choices=list(AMAZON_DICT.keys()), help="which Amazon reviews categories to download.")
    parser.add_argument('--limit', type=int, help='limit the number of reviews per json file')
    parser.add_argument('--verbose', default=True, help='whether or not to log progress of downloading to the console')
    args = parser.parse_args()

    if args.cat:
        download_amazon_reviews(args.cat, args.limit, args.verbose)
    else:
        with open(args.cat_spec, 'r') as file:
            cat = list(csv.reader(file, delimiter=',', skipinitialspace=True))[0]
        download_amazon_reviews(cat, args.limit, args.verbose)