##################################################
# Model training script.
##################################################

from models import *
from data_loader import ReviewsDataset, EssaysDataset
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import time
from sklearn import metrics

DEFAULT_MODEL_NAME = "model"
LOG_FILE_NAME = "log_results.csv"


def train(model, dataset, device, model_name, verbose=True, score_type='categorical', epochs=10, batch_size=1, lr=0.001, log_file=None):    
    # Set loss function to be compatible with the score type
    if(score_type in ['categorical', 'binary']):  loss_fun = CrossEntropyLoss().to(device)
    elif(score_type == 'standardized'):  loss_fun = MSELoss().to(device)
    else: raise NotImplementedError("No loss_fun chosen for this score_type")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = Adam(model.parameters(), lr=lr)
    model.train()
    
    if log_file:
        log_file.write("{m}\n".format(m=model_name))
        
    best_accuracy = 0.0
    for e in range(epochs):
        
        start = time.time()

        total_loss = 0.0
        num_correct = 0
        num_ex = 0
        pred_labels = np.array([])
        correct_labels = np.array([])
        for ex in data_loader:
            input_tensor, score = ex[0].to(device), ex[1].to(device)
            
            # make prediction
            pred = model(input_tensor)
            loss = loss_fun(pred, score)

            # backpropogate
            model.zero_grad()
            loss.backward()
            optim.step()

            # logging
            total_loss += loss.item()
            num_correct += (score == pred.argmax(dim=1)).sum()
            num_ex += len(score)
            pred_labels = np.append(pred_labels, pred.detach().to(torch.device('cpu')).argmax(dim=1).numpy())
            correct_labels = np.append(correct_labels, ex[1].detach().numpy())

        end = time.time()
        
        accuracy = np.round(100*float(num_correct) / num_ex, 2)
        best_accuracy = max(best_accuracy, accuracy)
        if verbose:
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            print("Epoch {ep} | \nloss: {l} -- accuracy: {a}% -- F1-score: {f}"
                  .format(ep=e, l=np.round(total_loss / len(dataset), 4), a=accuracy, f=metrics.f1_score(pred_labels, correct_labels, average='weighted')))
            print("Epoch duration: {t}s".format(t=np.round(end - start, 3)))
        if log_file:
            log_file.write("Epoch {ep} | \nloss: {l} -- accuracy: {a}% -- F1-score: {f}"
                  .format(ep=e, l=np.round(total_loss / len(dataset), 4), a=accuracy, f=metrics.f1_score(pred_labels, correct_labels, average='weighted')))
            log_file.write("Epoch duration: {t}s".format(t=np.round(end - start, 3)))            
        
    if log_file:
        log_file.write("Best accuracy: {a}\n".format(a=best_accuracy))
        
    save_model(model, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['reviews', 'essays'], default='reviews', help="Which dataset want to use.")
    parser.add_argument('-t', '--model_type', choices=['dan', 'transformer', 'best'], default='transformer', help="What type of model to use.")
    parser.add_argument('-f', '--model_file', help="The name of the model file to load for training  (loads %s<model_file>.pt)." % MODEL_DIR)
    parser.add_argument('-n', '--model_name', help="What name to give to the model  (will save <model_name>.pt after training is finished).")
    parser.add_argument('-tk', '--tokenizer', default='default', help="If 'default', will load pretrained tokenizer. Otherwise loads tokenizer of the given name (<tokenizer>.pt).")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="How many epochs to run the model for.")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="The batch size to use for training. Will pad / cut sequences if set to something other than '1'.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument('-s', '--score_type', choices=['categorical', 'binary', 'standardized'], default='categorical', help="The type of the scores. Be sure this is compatible with the model and dataset you want to use.")
    parser.add_argument('-sq', '--seq_len', type=int, help="The sequence length to use for training (in #tokens). Defaults to 1200 for essays and 300 for reviews. If batch size is >1 and seq len is set to 'max', will use the max len for dataset as seq_len.")
    
    parser.add_argument('-em', '--embed_dim', type=int, default=128, help="The embedding dimension.")
    parser.add_argument('-ah', '--attn_head', type=int, default=1, help="The number of attention heads to use.")
    parser.add_argument('-l', '--num_layers', type=int, default=1, help="The number of transformer layers to use.")
    parser.add_argument('-ffd', '--feedforward_dim', type=int, default=512, help="The internal feed forward dimension to use.")
    
    parser.add_argument('-v', '--verbose', action='store_false', help="Log training progress to the console.")
    parser.add_argument('--cpu_only', action='store_true', help="Only use the cpu during training")
    parser.add_argument('-log', '--log', action='store_true', help="Log model name and training accuracy to the log file.")

    args = parser.parse_args()

    if args.log:
        log_file = open(os.getcwd() + '/' + LOG_FILE_NAME, 'a')

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
            model = TransformerModel(dataset.vocab_size(), args.embed_dim, args.attn_head, args.num_layers, args.feedforward_dim, out_size=len(dataset.get_unique_labels()))
        else:
            raise Exception("Unknown model type: '{m}'".format(m=args.model_type))

    model = model.to(device)
    model_name = args.model_name if args.model_name else (args.model_file if args.model_file else DEFAULT_MODEL_NAME)
    print("Executing ...")
    train(model, dataset, device, model_name, verbose=args.verbose, score_type=args.score_type, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, log_file=log_file)

    if args.log:
        log_file.close()
