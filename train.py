##################################################
# Model training script.
##################################################

from models import *
from data_loader import ReviewsDataset, EssaysDataset
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

DEFAULT_MODEL_NAME = "model"


def train(model, dataset, model_name, verbose=True, score_type='categorical', epochs=10, batch_size=1, lr=0.001):
    # Set loss function to be compatible with the score type
    if(score_type in ['categorical', 'binary']):  loss_fun = CrossEntropyLoss()
    elif(score_type == 'standardized'):  loss_fun = MSELoss()
    else: raise NotImplementedError("No loss_fun chosen for this score_type")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = Adam(model.parameters(), lr=lr)
    model.train()

    # TODO:  (possibly) implement a validation loss. This will require splitting the data, though it's prone to sampling bias. For now just use training loss as metric.
    for e in range(epochs):
        total_loss = 0.0
        num_correct = 0
        num_ex = 0
        for ex in data_loader:
            # make prediction
            pred = model(ex[0])
            loss = loss_fun(pred, ex[1])

            # backpropogate
            model.zero_grad()
            loss.backward()
            optim.step()

            # logging
            total_loss += loss.item()
            num_correct += (ex[1] == pred.argmax(dim=1)).sum()
            num_ex += len(ex[1])

        if verbose:
            print("Epoch {ep}:    loss:   {l}      accuracy:   {a}%".format(ep=e, l=np.round(total_loss / len(dataset), 4), a=np.round(100*float(num_correct) / num_ex, 2)))
    save_model(model, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset want to use.")
    parser.add_argument('-t', '--model_type', choices=['dan', 'transformer'], default='dan', help="What type of model to use.")
    parser.add_argument('-f', '--model_file', help="The name of the model file to load for training  (loads %s<model_file>.pt)." % MODEL_DIR)
    parser.add_argument('-n', '--model_name', help="What name to give to the model  (will save <model_name>.pt after training is finished).")
    parser.add_argument('-tk', '--tokenizer', default='default', help="If 'default', will load pretrained tokenizer. Otherwise loads tokenizer of the given name (<tokenizer>.pt).")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="How many epochs to run the model for.")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="The batch size to use for training. Will pad / cut sequences if set to something other than '1'.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('-s', '--score_type', choices=['categorical', 'binary', 'standardized'], default='categorical', help="The type of the scores. Be sure this is compatible with the model and dataset you want to use.")
    parser.add_argument('-sq', '--seq_len', type=int, help="The sequence length to use for training (in #tokens). If batch size is greater than 1, will use 'max' for dataset seq_len.")
    parser.add_argument('-v', '--verbose', type=bool, default=True, help="Log training progress to the console.")

    args = parser.parse_args()

    print("Loading and preparing dataset ...")
    seq_len = args.seq_len if args.seq_len else ('max' if args.batch_size > 1 else None)
    dataset_args = {'score_type': args.score_type, 'seq_len': seq_len}
    if args.dataset == 'reviews':
        dataset = ReviewsDataset(**dataset_args) if args.tokenizer == 'default' else ReviewsDataset(args.tokenizer, **dataset_args)
    elif args.dataset == 'essays':
        dataset = EssaysDataset(**dataset_args) if args.tokenizer == 'default' else EssaysDataset(args.tokenizer, **dataset_args)
    else:  raise Exception("Unknown value for dataset: '{d}'".format(d=args.dataset))

    if args.model_file:
        model = load_model(args.model_file)
    else:
        if args.score_type == 'standardized':
            out_size = 1
        elif args.score_type in ['categorical', 'binary']:
            out_size = 2 if args.score_type == 'binary' else 5

        if args.model_type == 'dan':
            model = BasicDANModel(dataset.vocab_size(), out_size=out_size)
        elif args.model_type == 'transformer':
            model = TransformerModel(dataset.vocab_size(), out_size=out_size)
        else:
            raise Exception("Unknown model type: '{m}'".format(m=args.model_type))

    print("Executing ...")
    model_name = args.model_name if args.model_name else (args.model_file if args.model_file else DEFAULT_MODEL_NAME)
    train(model, dataset, model_name, verbose=args.verbose, score_type=args.score_type, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate)
