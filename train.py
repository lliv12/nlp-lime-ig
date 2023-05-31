'''
train.py

Main training script. Use this schema:

python train.py --dataset --model_type --model_file --model_name --tokenizer --epochs --batch_size --lr --class_weights --val_frac --cpu_only --verbose
                --embed_dim --attention_head --num_layers --feedforward_dim

  (Training params)
  --datset:  which dataset to train on ('reviews' or 'essays')
  --model_type:  the type of model to use for training (refer to models.py for module names)
  --model_file:  (optional) which model file to load and continue training on (Ex: 'transformer_model' (don't include the '.pt' extension))
  --tokenizer:  if not specified, will load pretrained tokenizer. Otherwise loads tokenizer of the given name (<tokenizer>.pt)
  --epochs:  how many epochs to run training for
  --batch_size:  batch size to use for training
  --lr:  learning rate to use for training
  --loss_func:  the loss function to use ('cross_entropy' or 'ordinal')
  --class_weights:  if specified, apply class imbalance weights for the loss function
  --val_frac:  fraction of the dataset to use for validation
  --cpu_only:  if specified, then only use the cpu for training
  --verbose:  log training progress to the console

  (Transformer params [only applies if using 'transformer' model])
  --embed_dim:  the embedding dimension
  --attention_head:  number of attention heads to use
  --num_enc_layers:  the number of encoder layers to use
  --feedforward_dim:  dimension of the encoder feedforward layers

Example:

python train.py --dataset reviews --model_type TransformerModel --model_name example_model --class_weights
'''

from data.dataset import ReviewsDataset, EssaysDataset, BucketSampler, padding_collate_func
from models import MODEL_DIR, load_model, save_model
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import argparse
from tqdm import tqdm

DEFAULT_MODEL_NAME = "model"


def ordinal_encode_label(labels, num_classes):
    batch_size = labels.size(0)
    encoded_labels = torch.zeros((batch_size, num_classes))
    class_indices = torch.arange(num_classes).unsqueeze(0)
    encoded_labels[torch.arange(batch_size).unsqueeze(1), class_indices] = (labels.unsqueeze(1) > class_indices).float()
    return encoded_labels

def pred_to_ordinal(pred):
    pred = sigmoid(pred)
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

class OrdinalLoss(nn.Module):
    def __init__(self, weight=None):
        super(OrdinalLoss, self).__init__()
        self.weight = weight.squeeze() if (weight != None) else None

    def forward(self, pred, labels):
        pred = sigmoid(pred)
        encoded_labels = ordinal_encode_label(labels, pred.size(-1))
        mse_loss = nn.MSELoss(reduction='none')(pred, encoded_labels).sum(axis=1)
        if self.weight != None:
            mse_loss = self.weight[labels] * mse_loss
        return mse_loss.sum()


def train(model, train_dataset, val_dataset, device, model_name, loss_func, verbose=True, epochs=10, batch_size=16, lr=0.001):
    train_loader = DataLoader(train_dataset, batch_sampler=BucketSampler(train_dataset, batch_size=batch_size), collate_fn=padding_collate_func)
    val_loader = DataLoader(val_dataset, batch_sampler=BucketSampler(val_dataset, batch_size=batch_size), collate_fn=padding_collate_func)
    optim = Adam(model.parameters(), lr=lr)
    model.train()
        
    best_accuracy = 0.0
    for e in range(epochs):
        if verbose:  print('\n' + 20*'-' + f" EPOCH {e} " + 20*'-' + '\n')

        total_loss = 0.0
        num_correct = 0
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

            if type(loss_func) == nn.CrossEntropyLoss:
                num_correct += (score == pred.argmax(dim=1)).sum().item()
            elif type(loss_func) == OrdinalLoss:
                total_loss += loss.item()
                pred_conv = pred_to_ordinal(pred)
                num_correct += (pred_conv == score).sum().item()

            num_examples = num_ex*batch_size
            accuracy = 100*(num_correct / num_examples)
            pbar.set_description(f"Avg Loss: {(total_loss / num_examples):.4f}   Correct: {num_correct} / {len(train_loader)*batch_size}  ({accuracy:.1f}%)")
            num_ex += 1

        num_correct = 0
        num_ex = 1
        pbar = tqdm(val_loader)
        model.eval()
        for ex in pbar:
            input_tensor, score = ex[0].to(device), ex[1].to(device)
            pred = model(input_tensor)

            if type(loss_func) == nn.CrossEntropyLoss:
                num_correct += (score == pred.argmax(dim=1)).sum().item()
            elif type(loss_func) == OrdinalLoss:
                pred_conv = pred_to_ordinal(pred)
                num_correct += (pred_conv == score).sum().item()

            num_examples = num_ex*batch_size
            accuracy = 100*(num_correct / num_examples)
            pbar.set_description(f"Correct: {num_correct} / {len(val_loader)*batch_size}  ({accuracy:.1f}%)")
            num_ex += 1
        if verbose:
            accuracy = 100*(num_correct / len(val_loader)*batch_size)
            if accuracy > best_accuracy:
                print(f"New best val accuracy:  {best_accuracy:.4f} ==> {accuracy:.4f}  |  Saving new model '{model_name}.pt' ...")
                best_accuracy = accuracy
                save_model(model, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset to use for training.")
    parser.add_argument('-t', '--model_type', default='TransformerModel', help="The type of model to use for training (refer to models.py for module names).")
    parser.add_argument('-f', '--model_file', help="The name of the model file to load for training  (loads %s<model_file>.pt)." % MODEL_DIR)
    parser.add_argument('-n', '--model_name', help="What name to give to the model  (will save <model_name>.pt checkpoint during training).")
    parser.add_argument('-tk', '--tokenizer', default='default', help="If 'default', will load pretrained tokenizer. Otherwise loads tokenizer of the given name (<tokenizer>.pt).")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="How many epochs to run the model for.")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="The batch size to use for training. Will pad / cut sequences if set to something other than '1'.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('--loss_func', default='cross_entropy', choices=['cross_entropy', 'ordinal'], help='The loss function to use')
    parser.add_argument('--class_weights', action='store_true', help='Whether or not to apply class imbalance weights for the loss function')
    parser.add_argument('--val_frac', default=0.2, type=float, help="The fraction of the dataset to use for validation")
    parser.add_argument('--cpu_only', action='store_true', help="Only use the cpu during training")
    parser.add_argument('-v', '--verbose', default=True, help="Log training progress to the console.")
    
    # transformer model options
    parser.add_argument('-em', '--embed_dim', type=int, default=128, help="The embedding dimension.")
    parser.add_argument('-ah', '--attn_head', type=int, default=2, help="The number of attention heads to use.")
    parser.add_argument('-l', '--num_enc_layers', type=int, default=2, help="The number of encoder layers to use.")
    parser.add_argument('-ffd', '--feedforward_dim', type=int, default=512, help="The internal feed forward dimension to use.")

    args = parser.parse_args()

    
    print("Loading and preparing %s dataset ..." % args.dataset)
    if args.dataset == 'reviews':
        dataset = ReviewsDataset()
    elif args.dataset == 'essays':
        dataset = EssaysDataset()
    else:  raise Exception("Unknown value for dataset: '{d}'".format(d=args.dataset))

    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using", device)

    if args.model_file:
        model = load_model(model_name=args.model_file)
    else:
        out_size = len(dataset.get_unique_labels())
        if args.model_type == 'BasicDANModel':
            model = load_model(model_type=args.model_type, vocab_size=dataset.vocab_size(), out_size=out_size)
        elif args.model_type == 'TransformerModel':
            model = load_model(model_type=args.model_type, vocab_size=dataset.vocab_size(), emb_dim=args.embed_dim, nhead=args.attn_head,
                               num_encoder_layers=args.num_enc_layers, dim_feedforward=args.feedforward_dim, out_size=out_size)
        else:
            model = load_model(model_type=args.model_type)

    train_dataset, val_dataset = random_split(dataset, [1-args.val_frac, args.val_frac])

    class_weights = train_dataset.dataset.compute_class_weights() if args.class_weights else None
    loss_func = nn.CrossEntropyLoss(weight=class_weights).to(device) if (args.loss_func == 'cross_entropy') else OrdinalLoss(weight=class_weights).to(device)

    model = model.to(device)
    model_name = args.model_name if args.model_name else (args.model_file if args.model_file else DEFAULT_MODEL_NAME)
    train(model, train_dataset, val_dataset, device, model_name, loss_func, verbose=args.verbose, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate)
