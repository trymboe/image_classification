import ssl
import torch
import argparse
import warnings
import numpy as np

from model import create_model
from utilities import get_statistics
from data_process import process_data
from training import train_model, evaluate


np.random.seed(42)
torch.manual_seed(42)

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('always')

CLASSES_LIST = ["forest", "buildings",
            "glacier", "street",
            "mountain", "sea"]

BATCH_SIZE = 128
LR = 0.001
MOMENTUM = 0.9
EPOCHS = 45




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image classification network')
    parser.add_argument('-t','--train', action='store_true')
    parser.add_argument('--model_path')

    parser.add_argument('-td','--test_data', action='store_true')
    parser.add_argument('--task2', action='store_true')
    parser.add_argument('--task3', action='store_true')


    args = vars(parser.parse_args())

    train = args['train']
    hooks = args['task2']
    task3 = args['task3']
    mac = False
    testdata = args['test_data']

    if mac:
        #for training on GPU for M1 mac
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train and not args['model_path']:
        print("if you want to train, you need to add a model path --model_path")
        exit()
    model_path = "models/model1_e50_lr0.001" if not train else args['model_path']

    train_loader, test_loader, val_loader = process_data("data", BATCH_SIZE, testdata=testdata)
    print("-- data loaded --")
    model, criterion, optimizer = create_model(device, LR, MOMENTUM)
    print("-- model created --")
    if train:
        print("-- training model --")
        train_model(model, criterion, optimizer, model_path, device, EPOCHS, train_loader, val_loader)
    else:
        print("-- loading model --")
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        if hooks or task3:
            print("-- calculating statistics --")
            get_statistics(model, device, task3)
        else:
            print("\n\n-- Test set evaluation --")
            loss, accuracy, class_wise_acc, class_wise_precision = evaluate(model, criterion, test_loader, device, show=False, test=True)