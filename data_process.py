import os
import torch
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataloader import Image_DL

np.random.seed(42)
torch.manual_seed(42)

CLASSES_LIST = ["forest", "buildings",
            "glacier", "street",
            "mountain", "sea"]

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

def process_data(path, batch_size, testdata = False):
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
        if "cifar" not in folder and "DS" not in folder and "imagenet" not in folder:
            full_path = path+'/'+folder+'/test' if testdata else path+'/'+folder
            for img in os.listdir(full_path):
                if img.endswith('.jpg') or img.endswith('.JPEG'):
                    image = np.array(Image.open(os.path.join(path,folder,img)))
                    image_hash = get_image_hash(image)
                    #Do not want duplicates or images of wrong dimention
                    if image.shape == (150,150,3) and image_hash not in hashes:
                        hashes.add(image_hash)
                        image = image.astype(np.float32) / 255.0

                        image = np.transpose(image, (2, 0, 1))

                        images.append(image)
                        labels.append(CLASSES_LIST.index(folder))

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



    train_loader = DataLoader(train_dl, batch_size=batch_size)
    test_loader = DataLoader(test_dl, batch_size=len(test_dl))
    val_loader = DataLoader(val_dl, batch_size=batch_size)

    return train_loader, test_loader, val_loader