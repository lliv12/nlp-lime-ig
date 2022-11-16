from models import MODEL_DIR, save_model, load_model, BasicDANModel
from data_loader import ReviewsDataset, EssaysDataset
from captum.attr import IntegratedGradients
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
from torch.optim import Adam
import argparse

DEFAULT_MODEL_NAME = "model"


def train(model, dataset, model_name, verbose=True, loss_type='mse', epochs=10, lr=0.001):
    # NOTE: the loss must be compatible with the format of the dataset labels
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['train', 'ig'], default='train')
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews')
    parser.add_argument('-t', '--model_type', choices=['dan'], default='dan')
    parser.add_argument('-f', '--model_file')
    parser.add_argument('-n', '--model_name')
    parser.add_argument('-e', '--epochs', type=int, default=10)

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
        train(model, dataset, model_name, epochs=args.epochs)
    elif args.mode == 'ig':
        inference(model)
    else:
        raise Exception("Unknown mode: '{m}'".format(m=args.mode))