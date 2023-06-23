'''
download_models.py

Module to download pretrained models. Refer to REVIEWS_MODELS for valid reviews models, ESSAYS_MODELS
for valid essays models

Schema:
python -m download_models.py --model
  --model:  name of the pretrained model you want to download (or 'all' to download all pretrained models)

Example:
python -m download_models.py --model transformer_reviews_5000
'''

import argparse
from huggingface_hub import hf_hub_url
import os
import requests
from models import MODEL_DIR, REVIEWS_SUBDIR, ESSAYS_SUBDIR

REVIEWS_MODELS = {
    'basic_transformer_reviews': {'REPO_ID': 'lliv12/basic_transformer_reviews', 'FILE': 'basic_transformer_reviews.pt'},
    'transformer_reviews_5000': {'REPO_ID': 'lliv12/transformer_reviews_5000', 'FILE': 'transformer_reviews_5000.pt'}
}
ESSAYS_MODELS = {
    'basic_transformer_essays': {'REPO_ID': 'lliv12/basic_transformer_essays', 'FILE': 'basic_transformer_essays.pt'},
    'transformer_essays_5000': {'REPO_ID': 'lliv12/transformer_essays_5000', 'FILE': 'transformer_essays_5000.pt'}
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', choices=(['all'] + list(REVIEWS_MODELS.keys()) + list(ESSAYS_MODELS.keys())), default='all')
    args = parser.parse_args()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    models = args.model if (args.model != 'all') else list(REVIEWS_MODELS.keys()) + list(ESSAYS_MODELS.keys())

    reviews_models = set(models).intersection(set(list(REVIEWS_MODELS.keys())))
    essays_models = set(models).intersection(set(list(ESSAYS_MODELS.keys())))
    
    if reviews_models:
        models_dir = os.path.join(MODEL_DIR, REVIEWS_SUBDIR)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        for model_name in reviews_models:
            print(f"Downloading model:  {model_name} ...")
            url = hf_hub_url(repo_id=REVIEWS_MODELS[model_name]['REPO_ID'], filename=REVIEWS_MODELS[model_name]['FILE'])
            response = requests.get(url)
            with open( os.path.join(models_dir, REVIEWS_MODELS[model_name]['FILE']) , "wb") as f:
                f.write(response.content)

    if essays_models:
        models_dir = os.path.join(MODEL_DIR, ESSAYS_SUBDIR)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        for model_name in essays_models:
            print(f"Downloading model:  {model_name} ...")
            url = hf_hub_url(repo_id=ESSAYS_MODELS[model_name]['REPO_ID'], filename=ESSAYS_MODELS[model_name]['FILE'])
            response = requests.get(url)
            with open( os.path.join(models_dir, ESSAYS_MODELS[model_name]['FILE']) , "wb") as f:
                f.write(response.content)

    print("Done")

