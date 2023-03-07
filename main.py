import os
import torch
import hashlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from resnet import ResNet18
from dataloader import Image_DL

np.random.seed(69)

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

BATCH_SIZE = 64
ACCUMULATION_STEPS = 4
LR = 0.001
MOMENTUM = 0.9
EPOCHS = 100


def get_image_hash(image):
    """
    Computes the SHA-256 hash of an image array.
    """
    return hashlib.sha256(image.tobytes()).hexdigest()

def check_for_duplicates(arrays):
    """
    Checks if any of the three arrays contain any of the exact same images.
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

    train_loader = DataLoader(train_dl, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dl, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dl, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, val_loader

def create_model(device):
    resnet18 = ResNet18(3, HEIGHT, WIDTH, len(CLASSES)).to(device)

    critirion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18.parameters(), lr=LR, momentum=MOMENTUM)
    return resnet18, critirion, optimizer

def train_model(model, critirion, optimizer, model_path, device):
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

        # Evaluate on validation set
        with torch.no_grad():
            val_loss, val_acc = evaluate(model, critirion, val_loader, device)
        
        print(f"Epoch {epoch+1}\nTraining loss: {running_loss/total:.4f} - Training accuracy {accuracy:.4f}\n"
              f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
    torch.save(model.state_dict(), model_path)

def evaluate(model,critirion, dataloader, device, show=False):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            loss = critirion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if show: #and idx % (num_batches/6) == 0:
                image = np.transpose(inputs[0], (2, 1, 0))
                plt.imshow(image)
                plt.title(f"correct {CLASSES_LIST[labels[0]]}, predicted {CLASSES_LIST[predicted[0]]}")
                plt.figure()

    accuracy = 100 * correct / total
    return loss, accuracy


if __name__ == "__main__":
    # Set device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #for training on GPU for M1 mac
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_path = "models/model1_100"
    train = True

    train_loader, test_loader, val_loader = process_data("data")
    print("data loaded")
    model, critirion, optimizer = create_model(device)
    print("model created")
    if train:
        train_model(model, critirion, optimizer, model_path, device)
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        loss, accuracy = evaluate(model, critirion, test_loader, device, show=True)
        print(f"Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}")
        plt.show()

    

