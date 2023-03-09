import os
import torch
import hashlib
import collections
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

from resnet import ResNet18
from dataloader import Image_DL

np.random.seed(69)
torch.manual_seed(69)

CLASSES = {
            "forest" : [1,0,0,0,0,0], "buildings" : [0,1,0,0,0,0],
            "glacier" : [0,0,1,0,0,0], "street" : [0,0,0,1,0,0],
            "mountain" : [0,0,0,0,1,0], "sea" : [0,0,0,0,0,1]
          }
CLASSES_LIST = ["forest", "buildings",
            "glacier", "street",
            "mountain", "sea"]
WIDTH = 150
HEIGHT = 150

BATCH_SIZE = 128
LR = 0.001
MOMENTUM = 0.9
EPOCHS = 2


def get_image_hash(image):
    """
    Computes the SHA-256 hash of an image array.
    
    Args:
    - image (numpy.ndarray)): Image that should be hashed

    Returns:
    - str: Hashed hexdecimals number
    """
    return hashlib.sha256(image.tobytes()).hexdigest()

def check_for_duplicates(arrays):
    """
    Checks if any of the three arrays contain any of the exact same images.

    Args:
    - arrays (list/numpy array): List/array of images

    Returns:
    - int: number of duplicates in the list
    """
    copies = 0
    hashes = set()
    for arr in arrays:
        for image in arr:
            image_hash = get_image_hash(image)
            if image_hash in hashes:
                copies += 1
                image = np.transpose(image, (2, 1, 0))
                plt.imshow(image)
                plt.figure
            hashes.add(image_hash)
    plt.show()
    return copies

def process_data(path):
    """
    Process image data located at `path` and split into train, validation, and test sets.
    Asserts that the splits are disjoint.
    
    Args:
    - path (str): The path to the directory containing the image data.
    
    Returns:
    - torch.utils.data.DataLoader: A DataLoader containing the training data.
    - torch.utils.data.DataLoader: A DataLoader containing the test data.
    - torch.utils.data.DataLoader: A DataLoader containing the validation data.
    """

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    images = []
    labels = []
    hashes = set()

    for folder in os.listdir(path):
        #dont want .DS_Store
        if "DS" not in folder:
            for img in os.listdir(path+'/'+folder):
                if img.endswith('.jpg'):
                    image = np.array(Image.open(os.path.join(path,folder,img)))
                    image_hash = get_image_hash(image)
                    #Do not want duplicates or images of wrong dimention
                    if image.shape == (150,150,3) and image_hash not in hashes:
                        hashes.add(image_hash)
                        image = image.astype(np.float32) / 255.0

                        image = np.transpose(image, (2, 0, 1))

                        images.append(image)
                        labels.append(CLASSES_LIST.index(folder))
                        # labels.append(CLASSES[folder])
            print(f"{folder} done")

    images = np.asarray(images)
    labels = np.asarray(labels)

    
    # Split data into train and validation sets
    x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

    # Split validation set into validation and test sets
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)


    
    assert check_for_duplicates([x_train, x_test, x_val]) == 0, "The obtained splits are not disjoint"

    print("assert done")


    train_dl = Image_DL(x_train, y_train)
    test_dl = Image_DL(x_test, y_test)
    val_dl = Image_DL(x_val, y_val)



    train_loader = DataLoader(train_dl, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dl, batch_size=len(test_dl))
    val_loader = DataLoader(val_dl, batch_size=len(val_dl))

    return train_loader, test_loader, val_loader

def create_model(device):
    '''
    Creates a ResNet18 PyTorch model for image classification.

    Args:
        device (torch.device): The device on which to create the model.

    Returns:
        - torch.nn.Module: A ResNet18 model.
        - torch.nn.CrossEntropyLoss: The loss function used for training the model.
        - torch.optim.Optimizer: The optimizer used for updating the model weights during training.
    '''
    resnet18 = ResNet18(3, HEIGHT, WIDTH, len(CLASSES), device).to(device)

    critirion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18.parameters(), lr=LR, momentum=MOMENTUM)
    return resnet18, critirion, optimizer

def train_model(model, critirion, optimizer, model_path, device):
    '''
    Trains a PyTorch model on a given dataset.
    
    Args:
    - model (torch.nn.Module): The model you want to train.
    - criterion (torch.nn.Module): The loss criterion you want to use.
    - optimizer (torch.optim.Optimizer): The optimizer you want to use.
    - dataloader (torch.utils.data.DataLoader): The dataloader containing the training data.
    - model_path (str): The path you want to save your model to.
    - device (torch.device): Device to use for training.
    '''
    training_loss = []
    validation_loss = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = critirion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}\nTraining loss: {running_loss/total:.4f} - Training accuracy {accuracy:.4f}\n")

        # Evaluate on validation set
        with torch.no_grad():
            val_loss, val_acc, _ = evaluate(model, critirion, val_loader, device, False)
        
        
        training_loss.append(running_loss/total)
        validation_loss.append(val_loss)

    x = np.arange(EPOCHS)

    plt.plot(x, training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig("training_loss.png")
    plt.figure()

    plt.plot(x, validation_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation loss")
    plt.savefig("validation_loss.png")


    with torch.no_grad():
        val_loss, val_acc, _ = evaluate(model, critirion, val_loader, device, True)

    torch.save(model.state_dict(), model_path)


def get_top_bottom_10(softmax, inputs, labels):
    # Get the indices of the highest and lowest probability scores
    max_values = torch.topk(softmax, k=1, dim=1)[0].squeeze(1)
    min_values = torch.topk(softmax, k=1, dim=1,largest=False)[0].squeeze(1)
    top_values, top_indices = torch.topk(max_values, k=10)
    bottom_values, bottom_indices = torch.topk(min_values, k=10, largest=False)

    for i in range(10):

        image = np.transpose(inputs[top_indices[i]].numpy(), (2, 1, 0))
        plt.imshow(image)
        plt.title(f"Top score: label: {CLASSES_LIST[labels[top_indices[i]]]}, predicted: {CLASSES_LIST[torch.topk(softmax[top_indices[0]], k=1)[1].item()]}")
        plt.figure()
        
        image = np.transpose(inputs[bottom_indices[i]].numpy(), (2, 1, 0))
        plt.imshow(image)
        plt.title(f"Bottom score: label: {CLASSES_LIST[labels[bottom_indices[i]]]}, predicted: {CLASSES_LIST[torch.topk(softmax[top_indices[0]], k=1)[1].item()]}")
        plt.figure()

    plt.show()
    



def evaluate(model, critirion, dataloader, device, per_class=True, show=False, test=False):
    '''
    Evaluates a dataset on a model. Is used for validation for validation set and test set.
    
    Args:
    - model (nn.Module): The model you want to evaluate
    - criterion (nn.Module): The criterion you want to use
    - dataloader (DataLoader): The dataloader object containing the data you want to evaluate
    - device (torch.device): The device you want to run the evaluation on
    - per_class (bool): If True, it will print the accuracy for each class individually
    - show (bool): If True, it will print one image from each batch with the prediction and label. Default False

    Returns:
    - float: Loss of the model
    - float: Accuracy of the model
    '''
    
    model.eval()
    correct = 0
    total = 0
    running_loss = 0

    softmax = 0

    correct_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    precision_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            loss = critirion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)          

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            print(collections.Counter(predicted.tolist()))
            print(collections.Counter(labels.tolist()))


            if test:
                get_top_bottom_10(outputs, inputs, labels)

            if per_class:
                #Calculate precision
                classwize_precision = precision_score(labels, predicted, average=None)
                average_precision = sum(classwize_precision) / len(classwize_precision)

                mask1 = torch.eq(labels, predicted)
                for i in range(len(CLASSES_LIST)):
                    total_counts[i] += (labels == i).sum().item()
                    
                    #Find the accuracy per class
                    mask2 = torch.eq(labels, i) 
                    mask3 = torch.eq(predicted, i)
                    mask4 = torch.logical_and(torch.logical_and(mask1, mask2), mask3)

                    correct_counts[i] += torch.sum(mask4 == True,dim=0)
                
                print("validation set:")
                print(f"Average accuracy is {100 * correct / total:.4f}")
                for i in range(len(CLASSES_LIST)):
                    accuracy = 100 * correct_counts[i] / total_counts[i]
                    print("Accuracy for class {} is {:.2f}%".format(CLASSES_LIST[i], accuracy))

                print(f"Average precision is {average_precision:.4f}")
                for i in range(len(CLASSES_LIST)):
                    print("Precision for class {} is {:.2f}".format(CLASSES_LIST[i], classwize_precision[i]))

                    
           
            if show:
                image = np.transpose(inputs[0], (2, 1, 0))
                plt.imshow(image)
                plt.title(f"correct {CLASSES_LIST[labels[0]]}, predicted {CLASSES_LIST[predicted[0]]}")
                plt.figure()


    val_loss = running_loss / total

    accuracy = 100 * correct / total

    return val_loss, accuracy, (softmax, labels)


if __name__ == "__main__":
    print("start")
    # Set device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #for training on GPU for M1 mac
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_path = "models/model1_150_lr=0.001"
    train = False

    train_loader, test_loader, val_loader = process_data("data")
    print("data loaded")
    model, critirion, optimizer = create_model(device)
    print("model created")
    if train:
        train_model(model, critirion, optimizer, model_path, device)
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        loss, accuracy, softmax = evaluate(model, critirion, test_loader, device, show=False, test=True)
        print(f"Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}")
        plt.show()

    

