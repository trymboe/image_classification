import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset



class Image_DL(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.tensor(labels)

         # calculate mean and standard deviation of the images
        mean = np.mean(images, axis=(0, 1, 2))
        std = np.std(images, axis=(0, 1, 2))
        
        # define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
