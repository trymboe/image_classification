import ssl
import torch
import numpy as np

from model import create_model
from utilities import get_statistics
from data_process import process_data
from training import train_model, evaluate


np.random.seed(42)
torch.manual_seed(42)

ssl._create_default_https_context = ssl._create_unverified_context

CLASSES_LIST = ["forest", "buildings",
            "glacier", "street",
            "mountain", "sea"]


BATCH_SIZE = 128
LR = 0.001
MOMENTUM = 0.9
EPOCHS = 5


if __name__ == "__main__":
    train = False
    hooks = True
    task3 = True
    mac = False

    if mac:
        #for training on GPU for M1 mac
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "models/model1_e45_lr0.001"

    train_loader, test_loader, val_loader = process_data("data", BATCH_SIZE, CLASSES_LIST)
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
        if hooks:
            print("-- calculating statistics --")
            get_statistics(model, device, test_loader, task3)
        else:
            print("\n\n-- Test set evaluation --")
            loss, accuracy, class_wise_acc, class_wise_precision = evaluate(model, criterion, test_loader, device, show=False, test=True)