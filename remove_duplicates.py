import os
import torch
import hashlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from PIL import Image

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
                plt.imshow(image)
                plt.figure()
                copies += 1
                # copies.append(image)
            hashes.add(image_hash)
    return copies



images = []
labels = []
hashes = set()
for folder in os.listdir("data"):
        #dont want .DS_Store
        if "DS" not in folder:
            for img in os.listdir("data"+'/'+folder):
                if img.endswith('.jpg'):
                    image = np.array(Image.open(os.path.join("data",folder,img)))
                    if image.shape == (150,150,3):
                        image_hash = get_image_hash(image)

                        if image_hash in hashes:
                            plt.imshow(image)
                            plt.figure()
                        hashes.add(image_hash)

                        # images.append(image)
                        # labels.append(0)


            print(f"{folder} done")



# x_train_val, x_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

# # Split validation set into validation and test sets
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, stratify=y_train_val)

# print(check_for_duplicates([x_train, x_test,x_val]))
plt.show()