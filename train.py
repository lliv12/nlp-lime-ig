##################################################
# Model training script.
##################################################

from models import *
from data_loader import ReviewsDataset, EssaysDataset
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
import numpy as np
import argparse

DEFAULT_MODEL_NAME = "model"


def train(model, dataset, model_name, verbose=True, score_type='categorical', epochs=10, lr=0.001):
    # Set loss function to be compatible with the score type
    if(score_type in ['categorical', 'binary']):  loss_fun = CrossEntropyLoss()
    elif(score_type == 'standardized'):  loss_fun = MSELoss()
    else: raise NotImplementedError("No loss_fun chosen for this score_type")
    optim = Adam(model.parameters(), lr=lr)
    model.train()
    for e in range(epochs):
        total_loss = 0.0
        for ex in dataset:
            # make prediction
            pred = model(ex[0])
            if score_type in ['categorical', 'binary']:  pred = pred.unsqueeze(dim=0)
            loss = loss_fun(pred, ex[1])

            # backpropogate
            model.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        if verbose:
            print("Epoch {ep} loss:   {l}".format(ep=e, l=total_loss / len(dataset)))
    save_model(model, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset want to use.")
    parser.add_argument('-t', '--model_type', choices=['dan', 'transformer'], default='dan', help="What type of model to use.")
    parser.add_argument('-f', '--model_file', help="The name of the model file to load for training  (loads %s<model_file>.pt)." % MODEL_DIR)
    parser.add_argument('-n', '--model_name', help="What name to give to the model  (will save <model_name>.pt after training is finished).")
    parser.add_argument('-tk', '--tokenizer', default='default', help="If 'default', will load pretrained tokenizer. Otherwise loads tokenizer of the given name (<tokenizer>.pt).")
    #parser.add_argument()
    parser.add_argument('-e', '--epochs', type=int, default=10, help="How many epochs to run the model for.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('-s', '--score_type', choices=['categorical', 'binary', 'standardized'], default='categorical', help="The type of the scores. Be sure this is compatible with the model and dataset you want to use.")
    parser.add_argument('-v', '--verbose', type=bool, default=True, help="Log training progress to the console.")

    args = parser.parse_args()

    print("Loading and preparing dataset ...")
    if args.dataset == 'reviews':
        score_type = args.score_type if args.score_type else 'categorical'
        dataset = ReviewsDataset(score_type=score_type) if args.tokenizer == 'default' else ReviewsDataset(args.tokenizer, score_type=score_type)
    elif args.dataset == 'essays':
        score_type = args.score_type if args.score_type else 'standardized'
        dataset = EssaysDataset(score_type=score_type) if args.tokenizer == 'default' else EssaysDataset(args.tokenizer, score_type=score_type)
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
    train(model, dataset, model_name, verbose=args.verbose, score_type=score_type, epochs=args.epochs, lr=args.learning_rate)
