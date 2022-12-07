##################################################
# Model training script.
##################################################

from models import *
from data_loader import ReviewsDataset, EssaysDataset
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import cProfile
import time

DEFAULT_MODEL_NAME = "model"


def train(model, dataset, device, model_name, verbose=True, score_type='categorical', epochs=10, batch_size=1, lr=0.001):    
    # Set loss function to be compatible with the score type
    if(score_type in ['categorical', 'binary']):  loss_fun = CrossEntropyLoss().to(device)
    elif(score_type == 'standardized'):  loss_fun = MSELoss().to(device)
    else: raise NotImplementedError("No loss_fun chosen for this score_type")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = Adam(model.parameters(), lr=lr)
    model.train()

    # TODO:  (possibly) implement a validation loss. This will require splitting the data, though it's prone to sampling bias. For now just use training loss as metric.
    for e in range(epochs):
        
        start = time.time()
        
        # with cProfile.Profile() as pr:
        
        total_loss = 0.0
        num_correct = 0
        num_ex = 0
        for ex in data_loader:
            input_tensor, score = ex[0].to(device), ex[1].to(device)
            
            # make prediction
            pred = model(input_tensor)
            loss = loss_fun(pred, score)
            # print(score.shape)
            # print(score)
            # print(input_tensor.shape)
            # print(input_tensor)
            # print(pred.shape)
            # print(pred)
            # print(loss)
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            # raise Exception("test")

            # backpropogate
            model.zero_grad()
            loss.backward()
            optim.step()

            # logging
            total_loss += loss.item()
            num_correct += (score == pred.argmax(dim=1)).sum()
            num_ex += len(score)

        end = time.time()
        
        if verbose:
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            print("Epoch {ep}:    loss:   {l}      accuracy:   {a}%".format(ep=e, l=np.round(total_loss / len(dataset), 4), a=np.round(100*float(num_correct) / num_ex, 2)))
            print("Epoch duration:", end - start)
        # pr.print_stats()
        # raise NotImplementedError()
    save_model(model, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset want to use.")
    parser.add_argument('-t', '--model_type', choices=['dan', 'transformer'], default='trans', help="What type of model to use.")
    parser.add_argument('-f', '--model_file', help="The name of the model file to load for training  (loads %s<model_file>.pt)." % MODEL_DIR)
    parser.add_argument('-n', '--model_name', help="What name to give to the model  (will save <model_name>.pt after training is finished).")
    parser.add_argument('-tk', '--tokenizer', default='default', help="If 'default', will load pretrained tokenizer. Otherwise loads tokenizer of the given name (<tokenizer>.pt).")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="How many epochs to run the model for.")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="The batch size to use for training. Will pad / cut sequences if set to something other than '1'.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('-s', '--score_type', choices=['categorical', 'binary', 'standardized'], default='categorical', help="The type of the scores. Be sure this is compatible with the model and dataset you want to use.")
    parser.add_argument('-sq', '--seq_len', type=int, help="The sequence length to use for training (in #tokens). Defaults to 1200 for essays and 300 for reviews. If batch size is >1 and seq len is set to 'max', will use the max len for dataset as seq_len.")
    parser.add_argument('-v', '--verbose', action='store_false', help="Log training progress to the console.")
    parser.add_argument('--cpu_only', action='store_true', help="Only use the cpu during training")

    args = parser.parse_args()

    print("Loading and preparing %s dataset ..." % args.dataset)
    if args.seq_len and args.seq_len == 'max':
        seq_len = 'max' if args.batch_size > 1 else None
    elif args.seq_len:
        seq_len = args.seq_len
    else:
        if args.dataset == 'essays':
            seq_len = 1200
        elif args.dataset == 'reviews':
            seq_len = 300
        else:
            seq_len = None
    
    dataset_args = {'score_type': args.score_type, 'seq_len': seq_len}
    if args.dataset == 'reviews':
        dataset = ReviewsDataset(**dataset_args) if args.tokenizer == 'default' else ReviewsDataset(args.tokenizer, **dataset_args)
    elif args.dataset == 'essays':
        dataset = EssaysDataset(**dataset_args) if args.tokenizer == 'default' else EssaysDataset(args.tokenizer, **dataset_args)
    else:  raise Exception("Unknown value for dataset: '{d}'".format(d=args.dataset))

    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using", device)

    if args.model_file:
        model = load_model(args.model_file)
    else:
        # if args.score_type == 'standardized':
        #     out_size = 1
        # elif args.score_type in ['categorical', 'binary']:
        #     out_size = 2 if args.score_type == 'binary' else len(dataset.get_unique_labels())

        if args.model_type == 'dan':
            model = BasicDANModel(dataset.vocab_size(), out_size=len(dataset.get_unique_labels()))
        elif args.model_type == 'transformer':
            model = TransformerModel(dataset.vocab_size(), out_size=len(dataset.get_unique_labels()))
        else:
            raise Exception("Unknown model type: '{m}'".format(m=args.model_type))

    model = model.to(device)
    model_name = args.model_name if args.model_name else (args.model_file if args.model_file else DEFAULT_MODEL_NAME)
    print("Executing ...")
    train(model, dataset, device, model_name, verbose=args.verbose, score_type=args.score_type, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate)
