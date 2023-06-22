'''
evaluate.py

Evaluate a saved model.

Schema:
python -m evaluate <model_name> <dataset> ... <options>
  + model_name:  name of the model
  + dataset:  which dataset to evaluate on ('reviews' or 'essays')
  --loss_func:  loss function used for training (allows evaluation to properly inference)
  --batch_size (-bs):  batch size for evaluation
  --metric:  'accuracy' or 'f1' (harmonic mean of f1 scores for each class)
  --vis_bar_chart:  if true, output a bar chart of per-class accuracies as a .png image
  --vis_conf_matrix:  if true, output a confusion matrix as a .png image
  --bar_chart_log_path:  where to save the bar chart image (will default to logs/<model_file>/bar_chart.png if not specified)
  --conf_mat_log_path:  where to save the confusion matrix image (will default to logs/<model_file>/conf_mat.png if not specified)
  --cpu_only:  only use the cpu for evaluation
  --num_workers:  number of threads to run for data loading
  --verbose:  print info to the console

Example:

python -m evaluate basic_transformer_essays essays --loss_func cross_entropy
'''

import argparse
import json
from tqdm import tqdm
from data.dataset import ReviewsDataset, EssaysDataset, BucketSampler, padding_collate_func
from losses.ordinal_loss import OrdinalLoss
from visualize import bar_chart, confusion_matrix
from torcheval.metrics.functional import multiclass_confusion_matrix
from torch.utils.data import DataLoader
from models import load_model
import torch
import torch.nn as nn
import os


TB_LOG_DIR = "logs"

def evaluate(model, model_name: str, dataset_type: str, device, dataloader, loss_func, batch_size: int = 16, save_metrics: bool = True, vis_bar_chart: bool = True, vis_conf_matrix: bool = True,
             bar_chart_log_path: bool = None, conf_mat_log_path: bool = None, verbose: bool = True):
    if isinstance(dataloader.dataset, torch.utils.data.dataset.Subset):
        dataset = dataloader.dataset.dataset
    elif isinstance(dataloader.dataset, torch.utils.data.Dataset):
        dataset = dataloader.dataset
    class_labels = list(range(1, dataset.num_classes+1))
    total_conf_mat = torch.zeros([dataset.num_classes, dataset.num_classes])
    num_ex = 1
    pbar = tqdm(dataloader)
    model.eval()
    for ex in pbar:
        input_tensor, score = ex[0].to(device), ex[1].to(device)
        pred = model(input_tensor)

        if type(loss_func) == nn.CrossEntropyLoss:
            pred = pred.argmax(dim=1)
        elif type(loss_func) == OrdinalLoss:
            pred = loss_func.pred_to_label(pred)
        else:
            raise Exception(f"Unknown loss function type: {loss_func}.")

        total_conf_mat += multiclass_confusion_matrix(pred, score, dataset.num_classes)
        label_correct = total_conf_mat.diag()
        label_count = total_conf_mat.sum(axis=-1)
        num_correct = label_correct.sum().int().item()

        # compute accuracy
        num_examples = num_ex*batch_size
        accuracy = 100*(num_correct / num_examples)
        # compute f1
        precision = label_correct / label_count
        recall = label_correct / label_count.sum()
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1 = f1_score.shape[0] / torch.sum(torch.reciprocal(f1_score[~torch.isnan(f1_score)]))

        pbar.set_description(f"Correct: {num_correct} / {len(dataloader)*batch_size}  (acc: {accuracy:.1f}%   f1: {f1:.3f})")
        num_ex += 1
    pbar.close()
    
    if vis_bar_chart:
        if (bar_chart_log_path == None):
            bar_chart_log_path = os.path.join(TB_LOG_DIR, dataset_type, model_name, 'eval')
            if verbose:  print(f"No bar chart save path provided.")
            if not os.path.exists(bar_chart_log_path):
                os.makedirs(bar_chart_log_path)
            bar_chart_log_path = os.path.join(bar_chart_log_path, 'bar_chart.png')
        if verbose:  print(f"Saving to: '{bar_chart_log_path}'")
        bar_chart(bar_chart_log_path, label_correct, label_count, class_labels)
    if vis_conf_matrix:
        if (conf_mat_log_path == None):
            conf_mat_log_path = os.path.join(TB_LOG_DIR, dataset_type, model_name, 'eval')
            if verbose:  print(f"No conf mat save path provided.")
            if not os.path.exists(conf_mat_log_path):
                os.makedirs(conf_mat_log_path)
            conf_mat_log_path = os.path.join(conf_mat_log_path, 'conf_mat.png')
        if verbose:  print(f"Saving to:  '{conf_mat_log_path}'")
        confusion_matrix(conf_mat_log_path, total_conf_mat, class_labels)

    metrics = {'accuracy': accuracy, 'f1': f1.item()}

    if save_metrics:
        log_dir = os.path.join(TB_LOG_DIR, dataset_type, model_name, 'eval')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str, help="Name of the model to load for evaluation  (loads %s<model_file>.pt)")
    parser.add_argument('dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset to evaluate on (will evaluate on the whole dataset)")
    parser.add_argument('--loss_func',  default='cross_entropy', choices=['cross_entropy', 'ordinal'], help='The loss function that was used for training (tells evaluation how to inference from model output)')
    parser.add_argument('--save_metrics', type=bool, default=True, help="Save metrics in a json file")
    parser.add_argument('--vis_bar_chart', type=bool, default=True, help="Output a bar chart of per-class accuracies as a .png image")
    parser.add_argument('--vis_conf_matrix', type=bool, default=True, help="Output a confusion matrix as a .png image")
    parser.add_argument('--bar_chart_log_path', type=str, help="where to save the bar chart image (will default to logs/<model_file>/eval/bar_chart.png if not specified)")
    parser.add_argument('--conf_mat_log_path', type=str, help="where to save the confusion matrix image (will default to logs/<model_file>/eval/conf_mat.png if not specified)")
    parser.add_argument('--cpu_only', action='store_true', help="Only use the cpu for evaluation")
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help="Batch size for the dataset")
    parser.add_argument('--num_workers', default=1, help='Number of threads to run for data loading')
    parser.add_argument('--verbose', type=bool, default=True, help="Print info to the console")
    args = parser.parse_args()

    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using", device)

    model_dict = load_model(model_name=os.path.join(args.dataset, args.model_file))
    model = model_dict['model']
    if args.dataset== 'reviews':
        dataset = ReviewsDataset(tokenizer=model_dict['tokenizer'])
    elif args.dataset == 'essays':
        dataset = EssaysDataset(tokenizer=model_dict['tokenizer'])
    
    dataloader = DataLoader(dataset, batch_sampler=BucketSampler(dataset, batch_size=args.batch_size), collate_fn=padding_collate_func, num_workers=args.num_workers)
    loss_func = nn.CrossEntropyLoss().to(device) if (args.loss_func == 'cross_entropy') else OrdinalLoss().to(device)

    result = evaluate(model, args.model_file, args.dataset, device, dataloader, loss_func, args.batch_size, args.save_metrics, args.vis_bar_chart, args.vis_conf_matrix,
                      args.bar_chart_log_path, args.conf_mat_log_path, args.verbose)
    print(result)