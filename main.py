import os
import ssl
import torch
import pickle
import hashlib
import statistics
import collections
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt



from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights

from resnet import ResNet18
from dataloader import Image_DL

np.random.seed(42)
torch.manual_seed(42)

ssl._create_default_https_context = ssl._create_unverified_context

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
EPOCHS = 45


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
    print("-- loading images --")
    for folder in os.listdir(path):
        #dont want .DS_Store
        if "cifar" not in folder and "DS" not in folder:
            for img in os.listdir(path+'/'+folder+'/test'):
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
    x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.176, stratify=labels, random_state=42)

    # Split validation set into validation and test sets
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.143, stratify=y_train_val, random_state=42)


    
    assert check_for_duplicates([x_train, x_test, x_val]) == 0, "The obtained splits are not disjoint"

    print("assert done")


    train_dl = Image_DL(x_train, y_train)
    test_dl = Image_DL(x_test, y_test)
    val_dl = Image_DL(x_val, y_val)



    train_loader = DataLoader(train_dl, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dl, batch_size=len(test_dl))
    val_loader = DataLoader(val_dl, batch_size=BATCH_SIZE)

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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18.parameters(), lr=LR, momentum=MOMENTUM)
    return resnet18, criterion, optimizer

def train_model(model, criterion, optimizer, model_path, device):
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

    total_CW_acc = []
    total_CW_prec = []
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            
        accuracy = 100 * correct / total

        print(f"\n\nEpoch {epoch+1}\nTraining accuracy {accuracy:.4f} - Training loss: {running_loss/total:.4f} ")

        # Evaluate on validation set
        with torch.no_grad():
            print("validation set:")
            val_loss, val_acc, class_wise_acc, class_wise_precision,  = evaluate(model, criterion, val_loader, device, False)
            class_wise_acc = [tensor[0].item() for tensor in class_wise_acc]
            total_CW_acc.append(class_wise_acc)
            total_CW_prec.append(class_wise_precision)

        
        training_loss.append(running_loss/total)
        validation_loss.append(val_loss)


    mean_acc = []
    mean_prec = []
    for i in range(EPOCHS):
        mean_acc.append(statistics.mean(total_CW_acc[i]))
        mean_prec.append(statistics.mean(total_CW_prec[i]))

    total_CW_acc = list(np.transpose(np.array(total_CW_acc)))
    total_CW_prec = list(np.transpose(np.array(total_CW_prec)))

    
    print(len(mean_acc), mean_acc)

    x = np.arange(EPOCHS)
    legend = ["forest", "buildings",
            "glacier", "street",
            "mountain", "sea", "mean"]

    for i in range(len(CLASSES_LIST)):
        plt.plot(x, total_CW_acc[i], linestyle="--")
    plt.plot(x, mean_acc, linewidth=2, color='red')
    plt.legend(legend)
    plt.title("Class-wise accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig("CW_accuracy.png")
    plt.figure()

    for i in range(len(CLASSES_LIST)):
        plt.plot(x, total_CW_prec[i], linestyle="--")
    plt.plot(x, mean_prec, linewidth=2, color='red')
    plt.legend(legend)
    plt.title("Class-wise precision")
    plt.xlabel("epochs")
    plt.ylabel("precision")
    plt.savefig("CW_precision.png")
    plt.figure()

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

    torch.save(model.state_dict(), model_path)

def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

def get_top_bottom_10(softmax, inputs, labels, class_nr):
    """Find the top-10 and bottom-10 images for a given class based on the softmax values, and display them in a plot.

    Args:
        softmax (torch.Tensor): A tensor of the softmax values for the six classes for each image.
        inputs (torch.Tensor): The input image as a tensor.
        labels (torch.Tensor): A tensor of the correct label for each image.
        class_nr (int): The number between 0 and 5, corresponding to the class we want to find the top-10 and bottom-10 images for.
    """

    _, preds = torch.max(softmax, dim=1)

    class_indices = (labels == class_nr).nonzero().squeeze()
    class_softmax = softmax[class_indices, class_nr]
    
    # Find the top-10 and bottom-10 images based on the softmax values
    top_indices = torch.argsort(class_softmax, descending=True)[:10]
    bottom_indices = torch.argsort(class_softmax, descending=False)[:10]
    
    fig, axs = plt.subplots(2, 10, figsize=(20, 4))
    
    # Display the top-10 images
    for i, idx in enumerate(top_indices):
        img = inputs[class_indices[idx]].permute(1, 2, 0).cpu().numpy()
        axs[0, i].imshow(img)
        axs[0, i].axis('off')
        axs[0, i].set_title(f"Pred: {CLASSES_LIST[preds[class_indices[idx]].item()]}")

    # Display the bottom-10 images
    for i, idx in enumerate(bottom_indices):
        img = inputs[class_indices[idx]].permute(1, 2, 0).cpu().numpy()
        axs[1, i].imshow(img)
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Pred: {CLASSES_LIST[preds[class_indices[idx]].item()]}")



    
def evaluate(model, criterion, dataloader, device, per_class=True, show=False, test=False):
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

    correct_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    precision_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    class_wise_precision = [0,0,0,0,0,0]

    class_wise_acc = [[],[],[],[],[],[]]

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)          

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #Calculate precision
            class_wise_precision += precision_score(labels.cpu(), predicted.cpu(), average=None)
            #Calculating class-wise accuracy
            mask1 = torch.eq(labels, predicted)
            for i in range(len(CLASSES_LIST)):
                total_counts[i] += (labels == i).sum().item()
                
                #Find the accuracy per class
                mask2 = torch.eq(labels, i) 
                mask3 = torch.eq(predicted, i)
                mask4 = torch.logical_and(torch.logical_and(mask1, mask2), mask3)

                correct_counts[i] += torch.sum(mask4 == True,dim=0)
                
            if test:
                for i in range(3):
                    get_top_bottom_10(outputs, inputs, labels, i)
                plt.show()

                torch.save(outputs, 'test_softmax.pt')
                torch.save(labels, 'test_labels.pt')

           
            if show:
                image = np.transpose(inputs[0], (2, 1, 0))
                plt.imshow(image)
                plt.title(f"correct {CLASSES_LIST[labels[0]]}, predicted {CLASSES_LIST[predicted[0]]}")
                plt.figure()

        average_precision = sum(class_wise_precision) / len(class_wise_precision)
        
        print(f"Average accuracy is {100 * correct / total:.4f}, loss is {running_loss/total}")
        print(f"class-wise accuracy is:", end=" ")
        for i in range(len(CLASSES_LIST)):
            accuracy = 100 * correct_counts[i] / total_counts[i]
            class_wise_acc[i].append(accuracy)
            print("{} : {:.2f}%".format(CLASSES_LIST[i], accuracy), end=" - ")

        print(f"\nAverage precision is {average_precision:.4f}")
        print(f"class-wise precision is:", end=" ")
        for i in range(len(CLASSES_LIST)):
            print("{} : {:.2f}".format(CLASSES_LIST[i], class_wise_precision[i]), end=" - ")



    val_loss = running_loss / total

    accuracy = 100 * correct / total

    return val_loss, accuracy, class_wise_acc, class_wise_precision

def get_statistics(model, device, dataloader, task3):
    all_modules = []
    for nam, mod in model.named_modules():
        if "conv" in nam:
            all_modules.append(mod)

    #I use the 2 first and 3 last moduelsed with conv in the name
    first_modules = all_modules[:2]
    last_modules = all_modules[-3:]
    selected_modules = first_modules + last_modules


    hooks = []
    for module in selected_modules:
        if task3:
            hook = module.register_forward_hook(forward_hook1)
        else:
            hook = module.register_forward_hook(forward_hook2)    
        hooks.append(hook)
  
    if task3:
        resnet_model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        datasets = [dataloader]
        cifar = unpickle("data/cifar-10-batches-py/test_batch")




    else:
        # Iterate over your dataloader and compute the average non-positive percentage for each feature map
        count = 0
        total_non_pos_percentages = [0] * 5
        for inputs, targets in dataloader:
            outputs = model(inputs.to(device))
            count += 1
            if count == 200:
                break

    # Remove the forward hooks
    for hook in hooks:
        hook.remove()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def forward_hook1(module, input, output):
    non_pos_count = torch.sum(output <= 0)
    total_count = output.numel()
    non_pos_percentage = non_pos_count.float() / total_count * 100
    avg_non_pos_percentage = non_pos_percentage / 200
    print(f"Average non-positive percentage in {module} output: {avg_non_pos_percentage:.2f}%")

def forward_hook2(module, input, output):
    k = 1000
    feature_map = output.detach()
    spatial_mean = torch.mean(feature_map, dim=[2, 3], keepdim=True)
    # Reshape the mean feature map tensor to (batch_size, num_channels), assuming num_channels is the lastdimention
    mean_feature_map = spatial_mean.view(spatial_mean.size(0), -1)
    covariance_matrix = torch.cov(mean_feature_map)

    # Compute the top-k eigenvalues and sort them in decreasing order
    eigenvalues, _ = torch.symeig(covariance_matrix, eigenvectors=False)
    top_k_eigenvalues, _ = torch.sort(eigenvalues, descending=True)[:k]

    plt.bar(range(len(top_k_eigenvalues)), top_k_eigenvalues)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue magnitude')
    plt.title('Top-k eigenvalues of empirical covariance matrix')
    plt.show()




if __name__ == "__main__":
    # Set device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #for training on GPU for M1 mac
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_path = "models/model1_e45_lr0.001"
    train = False
    hooks = False
    task3 = False

    

    train_loader, test_loader, val_loader = process_data("data")
    print("-- data loaded --")
    model, criterion, optimizer = create_model(device)
    print("-- model created --")
    if train:
        print("-- training model --")
        train_model(model, criterion, optimizer, model_path, device)
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