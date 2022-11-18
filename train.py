from models import MODEL_DIR, save_model, load_model, BasicDANModel
from data_loader import ReviewsDataset, EssaysDataset
from captum.attr import IntegratedGradients
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
from torch.optim import Adam
import numpy as np
import argparse

DEFAULT_MODEL_NAME = "model"


def train(model, dataset, model_name, verbose=True, score_type='categorical', epochs=10, lr=0.001):
    # Set loss function to be compatible with the score type
    if(score_type == 'categorical'):  loss_fun = CrossEntropyLoss()
    elif(score_type == 'binary'):  loss_fun = BCELoss()
    elif(score_type == 'standardized'):  loss_fun = MSELoss()
    optim = Adam(model.parameters(), lr=lr)
    model.train()
    for e in range(epochs):
        total_loss = 0.0
        for ex in dataset:
            # make prediction
            pred = model(ex[0])
            if score_type == 'categorical':  pred = pred.unsqueeze(dim=0)
            loss = loss_fun(pred, ex[1])

            # backpropogate
            model.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        if verbose:
            print("Epoch {ep} loss:   {l}".format(ep=e, l=total_loss / len(dataset)))
    save_model(model, model_name)


def inference(model, dataset, ex_id=None):
    # NOTE: Does not appear to work with scalar target (predicting standardized scores)
    # https://captum.ai/api/integrated_gradients.html
    # https://github.com/pytorch/captum/issues/405
    model.eval()
    if not ex_id:  ex_id = np.random.randint(len(dataset))
    ig = IntegratedGradients(model.forward_emb)
    attributions = ig.attribute(inputs=model.get_embeddings(dataset[ex_id][0]), baselines=None, target=dataset[ex_id][1].long().item())
    scores = np.mean(attributions.detach().numpy(), axis=2).squeeze()
    print("SCORES: ", scores)
    print("SHAPE: ", scores.shape)

    # TODO: Add visualization of token scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['train', 'ig'], default='train', help="What mode to run on.")
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset to use for training / evaluation.")
    parser.add_argument('-t', '--model_type', choices=['dan'], default='dan', help="What type of model to use for training.")
    parser.add_argument('-f', '--model_file', help="The name of the model file to load for training / evaluation  (loads {model_file}.pt).")
    parser.add_argument('-n', '--model_name', help="What name to give to the model  (will save {model_name}.pt after training is finished).")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="How many epochs to run the model for.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('-s', '--score_type', choices=['categorical', 'binary', 'standardized'], help="The type of the scores. Be sure this is compatible with the model and dataset you want to use.")
    parser.add_argument('-ex_id', '--ex_id', help="The example id you want to make inference on (will be chosen randomly if not specified)")
    parser.add_argument('-v', '--verbose', default=True, help="Log training progress to the console.")

    args = parser.parse_args()

    print("Loading and preparing dataset ...")
    if args.dataset == 'reviews':
        score_type = args.score_type if args.score_type else 'categorical'
        dataset = ReviewsDataset(score_type=score_type)
    elif args.dataset == 'essays':
        score_type = args.score_type if args.score_type else 'standardized'
        dataset = EssaysDataset(score_type=score_type)
    else:  raise Exception("Unknown value for dataset: '{d}'".format(d=args.dataset))

    if args.model_file:
        model = load_model(args.model_file)
    elif args.model_type == 'dan':
        out_size = 1 if args.dataset == 'essays' else 5
        model = BasicDANModel(dataset.vocab_size(), out_size=out_size, bin=args.score_type == 'binary')
    else:
        raise Exception("Unknown model type: '{m}'".format(m=args.model_file))

    print("Executing ...")
    if args.mode == 'train':
        model_name = args.model_name if args.model_name else (args.model_file if args.model_file else DEFAULT_MODEL_NAME)
        train(model, dataset, model_name, verbose=args.verbose, score_type=score_type, epochs=args.epochs, lr=args.learning_rate)
    elif args.mode == 'ig':
        inference(model, dataset, args.ex_id)
    else:
        raise Exception("Unknown mode: '{m}'".format(m=args.mode))