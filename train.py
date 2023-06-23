'''
train.py

Main training script. Use this schema:

python -m train --dataset --model_type ... <options>

  (Training params)
  --datset:  which dataset to train on ('reviews' or 'essays')
  --model_type:  the type of model to use for training (refer to models.py for module names)
  --model_file:  (optional) which model file to load and continue training on (Ex: 'transformer_model' (don't include the '.pt' extension))
  --model_name:  What name to give to the model  (will save into <dataset>/<model_name>.pt checkpoint during training).
  --tokenizer:  if specified, will load this pretrained tokenizer instead of the default for this dataset (unless it's not found or a model file is specified (whose tokenizer will take precedence))
  --epochs:  how many epochs to run training for
  --batch_size:  batch size to use for training
  --lr:  learning rate to use for training
  --loss_func:  the loss function to use ('cross_entropy' or 'ordinal')
  --class_weights:  if specified, apply class imbalance weights for the loss function; helps to counteract class imbalance
  --undersample:  if specified, undersample majority classes during training; helps to counteract class imbalance
  --oversample:  if specified, oversample minority classes during training; helps to counteract class imbalance
  --val_frac:  fraction of the dataset to use for validation
  --cpu_only:  if specified, then only use the cpu for training
  --num_workers:  number of threads to run for data loading in the training loop
  --clear_tb_logs:  clear previous tb logs for this model before starting the next training run
  --tb_loss_interval:  log training loss to tensorboard every N global steps
  --tb_vis_interval:  log bar chart and confusion matrix to tensorboard every N global steps
  --metric:  the metric to use for evaluation (save the best model based on this metric)
  --verbose:  log training progress to the console

  (Transformer params [only applies if using 'transformer' model])
  --embed_dim (-em):  the embedding dimension
  --attention_head (-ah):  number of attention heads to use
  --num_enc_layers (-l):  the number of encoder layers to use
  --feedforward_dim (-ffd):  dimension of the encoder feedforward layers

Example:

[Train Amazon reviews model  (config for basic_transformer_reviews)]:
python -m train --dataset reviews --model_type TransformerModel --model_name basic_transformer_reviews_2 --undersample --metric f1 --tb_vis_interval 150

[Train Essays model  (config for basic_transformer_essays)]:
python -m train --dataset essays --model_type TransformerModel --model_name basic_transformer_essays_2 --undersample --metric f1 --loss_func ordinal --tb_vis_interval 150
'''

import os
import shutil
from data.dataset import ReviewsDataset, EssaysDataset, BucketSampler, padding_collate_func
from models import MODEL_DIR, load_model, save_model
from losses.ordinal_loss import OrdinalLoss
from evaluate import evaluate
from visualize import bar_chart, confusion_matrix
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.image as mpimg
from torcheval.metrics.functional import multiclass_confusion_matrix

DEFAULT_MODEL_NAME = "model"
TB_LOG_DIR = "logs"
BARCHART_BASEFILENAME = "bar_chart.png"
CONFUSION_MAT_BASEFILENAME = "conf_mat.png"


def clear_logs(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    train_subdir = os.path.join(dir, 'train')
    val_subdir = os.path.join(dir, 'val')
    if os.path.exists(train_subdir):  shutil.rmtree(train_subdir)
    if os.path.exists(val_subdir):  shutil.rmtree(val_subdir)

def log_img_to_tb(tb_path, im_path, tb_writer, step):
    im_tensor = torch.from_numpy( mpimg.imread(im_path) ).permute(2, 0, 1)
    tb_writer.add_image(tb_path, im_tensor, step, dataformats='CHW')

def train(model, tokenizer_name, train_dataset, val_dataset, device, model_name, loss_func, metric='accuracy', verbose=True, epochs=10, batch_size=16, lr=0.001, num_workers=0,
          clear_tb_logs=True, tb_loss_interval=10, tb_vis_interval=50, undersample=False, oversample=False):
    if undersample and oversample:
        raise Exception("Specify either 'undersample' or 'oversample', but not both.")
        
    train_loader = DataLoader(train_dataset, batch_sampler=BucketSampler(train_dataset, batch_size=batch_size, undersample=undersample, oversample=oversample), collate_fn=padding_collate_func, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_sampler=BucketSampler(val_dataset, batch_size=batch_size), collate_fn=padding_collate_func, num_workers=num_workers)
    optim = Adam(model.parameters(), lr=lr)

    tb_dir = os.path.join(TB_LOG_DIR, model_name)
    if clear_tb_logs and os.path.exists(tb_dir):
        clear_logs(tb_dir)
    tb_writer = SummaryWriter(os.path.join(TB_LOG_DIR, model_name))
    class_labels = list(range(1, train_dataset.dataset.num_classes+1))

    base_train_logs_dir = os.path.join(tb_dir, 'train')
    base_val_logs_dir = os.path.join(tb_dir, 'val')
    train_bar_chart_path = os.path.join(base_train_logs_dir, BARCHART_BASEFILENAME)
    val_bar_chart_path = os.path.join(base_val_logs_dir, BARCHART_BASEFILENAME)
    train_conf_mat_path = os.path.join(base_train_logs_dir, CONFUSION_MAT_BASEFILENAME)
    val_conf_mat_path = os.path.join(base_val_logs_dir, CONFUSION_MAT_BASEFILENAME)
    os.makedirs(base_train_logs_dir)
    os.makedirs(base_val_logs_dir)

    global_step = 0

    model.train()

    best_score = 0.0
    for e in range(epochs):
        if verbose:  print('\n' + 20*'-' + f" EPOCH {e} " + 20*'-' + '\n')

        total_loss = 0.0
        total_conf_mat = torch.zeros([train_dataset.dataset.num_classes, train_dataset.dataset.num_classes])
        num_ex = 1
        pbar = tqdm(train_loader)
        model.train()
        for ex in pbar:
            input_tensor, score = ex[0].to(device), ex[1].to(device)
            
            pred = model(input_tensor)
            loss = loss_func(pred, score)

            model.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if type(loss_func) == nn.CrossEntropyLoss:
                pred = pred.argmax(dim=1)
            elif type(loss_func) == OrdinalLoss:
                pred = loss_func.pred_to_label(pred)
            else:
                raise Exception(f"Unknown loss function type: {loss_func}.")

            total_conf_mat += multiclass_confusion_matrix(pred, score, train_dataset.dataset.num_classes)
            label_correct = total_conf_mat.diag()
            label_count = total_conf_mat.sum(axis=-1)
            num_correct = label_correct.sum().int().item()

            num_examples = num_ex*batch_size
            accuracy = 100*(num_correct / num_examples)
            pbar.set_description(f"Avg Loss: {(total_loss / num_examples):.4f}   Correct: {num_correct} / {len(train_loader)*batch_size}  ({accuracy:.1f}%)")
            if global_step % tb_loss_interval == 0:
                tb_writer.add_scalar('train/loss', loss.item(), global_step)
            if global_step > 0 and global_step % tb_vis_interval == 0:
                bar_chart(train_bar_chart_path, label_correct, label_count, class_labels)
                confusion_matrix(train_conf_mat_path, total_conf_mat, class_labels)
                log_img_to_tb('train/Class-level Metrics/bar_chart', train_bar_chart_path, tb_writer, global_step)
                log_img_to_tb('train/Class-level Metrics/conf_mat', train_conf_mat_path, tb_writer, global_step)
            num_ex += 1

            global_step += 1
        pbar.close()

        log_vis = tb_vis_interval > 0
        metrics = evaluate(model, model_name, val_dataset.dataset.type, device, val_loader, loss_func, batch_size=batch_size, vis_bar_chart=log_vis, vis_conf_matrix=log_vis,
                            bar_chart_log_path=val_bar_chart_path, conf_mat_log_path=val_conf_mat_path, verbose=False)

        if metrics[metric] > best_score:
            if verbose:  print(f"New best val {metric}:  {best_score:.4f} ==> {metrics[metric]:.4f}  |  Saving new model '{model_name}.pt' ...")
            best_score = metrics[metric]
            save_model(model, model_name, tokenizer_name)
        tb_writer.add_scalar(f'val/{metric}', metrics[metric], e)
        if tb_vis_interval > 0:
            log_img_to_tb('val/Class-level Metrics/bar_chart', val_bar_chart_path, tb_writer, e)
            log_img_to_tb('val/Class-level Metrics/conf_mat', val_conf_mat_path, tb_writer, e)

    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset to use for training.")
    parser.add_argument('-t', '--model_type', default='TransformerModel', help="The type of model to use for training (refer to models.py for module names).")
    parser.add_argument('-f', '--model_file', help="The name of the model file to load for training  (loads %s<model_file>.pt)." % MODEL_DIR)
    parser.add_argument('-n', '--model_name', help="What name to give to the model  (will save into <dataset>/<model_name>.pt checkpoint during training).")
    parser.add_argument('-tk', '--tokenizer', type=str, help="If specified, will load this pretrained tokenizer instead of the default for this dataset (unless it's not found or a model file is specified (whose tokenizer will take precedence))")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="How many epochs to run the model for.")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="The batch size to use for training. Will pad / cut sequences if set to something other than '1'.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('--loss_func', default='cross_entropy', choices=['cross_entropy', 'ordinal'], help='The loss function to use')
    parser.add_argument('--class_weights', action='store_true', help='Whether or not to apply class imbalance weights for the loss function; helps to counteract class imbalance')
    parser.add_argument('--undersample', action='store_true', help="Undersample majority classes during training; helps to counteract class imbalance")
    parser.add_argument('--oversample', action='store_true', help="Oversample minority classes during training; helps to counteract class imbalance")
    parser.add_argument('--val_frac', default=0.2, type=float, help="The fraction of the dataset to use for validation")
    parser.add_argument('--cpu_only', action='store_true', help="Only use the cpu during training")
    parser.add_argument('--num_workers', default=0, help='Number of threads to run for data loading in the training loop')
    parser.add_argument('--clear_tb_logs', default=True, help='clear previous tb logs for this model before starting the next training run')
    parser.add_argument('--tb_loss_interval', default=10, help='Log progress to tensorboard every N steps.')
    parser.add_argument('--tb_vis_interval', default=-float('inf'), type=int, help='Log bar chart and confusion matrix to tensorboard every N steps.')
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'f1'], help="The metric to use for evaluation (save the best model based on this metric)")
    parser.add_argument('-v', '--verbose', default=True, help="Log training progress to the console.")
    
    # transformer model options
    parser.add_argument('-em', '--embed_dim', type=int, default=128, help="The embedding dimension.")
    parser.add_argument('-ah', '--attn_head', type=int, default=2, help="The number of attention heads to use.")
    parser.add_argument('-l', '--num_enc_layers', type=int, default=2, help="The number of encoder layers to use.")
    parser.add_argument('-ffd', '--feedforward_dim', type=int, default=512, help="The internal feed forward dimension to use.")

    args = parser.parse_args()

    if args.undersample and args.oversample:
        raise Exception("Specify either 'undersample' or 'oversample', but not both.")

    tokenizer = args.tokenizer

    if args.model_file:
        model_dict = load_model(model_name=args.model_file)
        model = model_dict['model']
        tokenizer = model_dict['tokenizer']
    
    print("Loading and preparing %s dataset ..." % args.dataset)
    if args.dataset == 'reviews':
        dataset = ReviewsDataset(tokenizer=tokenizer)
    elif args.dataset == 'essays':
        dataset = EssaysDataset(tokenizer=tokenizer)
    else:  raise Exception("Unknown value for dataset: '{d}'".format(d=args.dataset))

    if not args.model_file:
        out_size = len(dataset.get_unique_labels())
        if args.model_type == 'BasicDANModel':
            model = load_model(model_type=args.model_type, vocab_size=dataset.vocab_size(), out_size=out_size)
        elif args.model_type == 'TransformerModel':
            model = load_model(model_type=args.model_type, vocab_size=dataset.vocab_size(), emb_dim=args.embed_dim, nhead=args.attn_head,
                               num_encoder_layers=args.num_enc_layers, dim_feedforward=args.feedforward_dim, out_size=out_size)
        else:
            model = load_model(model_type=args.model_type)

    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using", device)

    train_dataset, val_dataset = random_split(dataset, [1-args.val_frac, args.val_frac])

    class_weights = train_dataset.dataset.compute_class_weights() if args.class_weights else None
    loss_func = nn.CrossEntropyLoss(weight=class_weights).to(device) if (args.loss_func == 'cross_entropy') else OrdinalLoss(weight=class_weights).to(device)

    model = model.to(device)
    model_name = args.model_name if args.model_name else (args.model_file if args.model_file else DEFAULT_MODEL_NAME)
    model_path = os.path.join(args.dataset, model_name)
    train(model, tokenizer, train_dataset, val_dataset, device, model_path, loss_func, metric=args.metric, verbose=args.verbose, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.learning_rate, num_workers=args.num_workers, clear_tb_logs=args.clear_tb_logs, tb_loss_interval=args.tb_loss_interval,
          tb_vis_interval=args.tb_vis_interval, undersample=args.undersample, oversample=args.oversample)
