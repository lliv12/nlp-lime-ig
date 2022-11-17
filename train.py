from models import MODEL_DIR, save_model, load_model, BasicDANModel
from data_loader import ReviewsDataset, EssaysDataset
from captum.attr import IntegratedGradients
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
from torch.optim import Adam
import argparse

DEFAULT_MODEL_NAME = "model"


def train(model, dataset, model_name, verbose=True, loss_type='mse', epochs=10, lr=0.001):
    # NOTE: the loss must be compatible with the network output layer (and incoming labels of course)
    if(loss_type == 'mse'):  loss_fun = MSELoss()
    elif(loss_type == 'cat'):  loss_fun = CrossEntropyLoss()
    elif(loss_type == 'bin'):  loss_fun = BCELoss()
    else:  raise Exception("Unknown loss_type:  '{l}'".format(l=loss_type))
    optim = Adam(model.parameters(), lr=lr)
    model.train()
    for e in range(epochs):
        total_loss = 0.0
        for ex in dataset:
            # make prediction
            pred = model(ex[0])
            loss = loss_fun(pred, ex[1])

            # backpropogate
            model.zero_grad()
            loss.backward()
            optim.step()

            # logging
            total_loss += loss.item()
        if verbose:
            print("Epoch {ep} loss:   {l}".format(ep=e, l=total_loss / len(dataset)))
    save_model(model, model_name)


def inference(model):
    model.eval()
    raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['train', 'ig'], default='train', help="What mode to run on.")
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset to use for training / evaluation.")
    parser.add_argument('-t', '--model_type', choices=['dan'], default='dan', help="What type of model to use for training.")
    parser.add_argument('-f', '--model_file', help="The name of the model file to load for training / evaluation  (loads {model_file}.pt).")
    parser.add_argument('-n', '--model_name', help="What name to give to the model  (will save {model_name}.pt after training is finished).")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="How many epochs to run the model for.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('-l', '--loss_type', default='mse', help="The loss to use for training the model. Be sure this is compatible with the model you want to train.")
    parser.add_argument('-v', '--verbose', default=True, help="Log training progress to the console.")

    args = parser.parse_args()

    if args.dataset == 'reviews':  dataset = ReviewsDataset()
    elif args.dataset == 'essays':  dataset = EssaysDataset(score_type='standardized')
    else:  raise Exception("Unknown value for dataset: '{d}'".format(d=args.dataset))

    if args.model_file:
        model = load_model(args.model_file)
    elif args.model_type == 'dan':
        model = BasicDANModel(dataset.vocab_size())
    else:
        raise Exception("Unknown model type: '{m}'".format(m=args.model_file))

    if args.mode == 'train':
        model_name = args.model_name if args.model_name else (args.model_file if args.model_file else DEFAULT_MODEL_NAME)
        train(model, dataset, model_name, verbose=args.verbose, loss_type=args.loss_type, epochs=args.epochs, lr=args.learning_rate)
    elif args.mode == 'ig':
        inference(model)
    else:
        raise Exception("Unknown mode: '{m}'".format(m=args.mode))