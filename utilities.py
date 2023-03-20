import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from dataloader import Image_DL
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torchvision.models import resnet18, ResNet18_Weights

from data_process import process_data

def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

def get_statistics(model, device, task3):
    all_modules = []
    for nam, mod in model.named_modules():
        if "conv" in nam:
            all_modules.append(mod)

    #I use the 5 last moduelsed with conv in the name
    selected_modules = all_modules[-5:]


    if task3:
        resnet_model = resnet18(weights = ResNet18_Weights.DEFAULT).to(device)

        for nam, mod in resnet_model.named_modules():
            if "conv" in nam:
                all_modules.append(mod)

        #I use the 2 first and 3 last moduelsed with conv in the name
        selected_modules = all_modules[-5:]
        
        hooks = []
        for module in selected_modules:
            hook = module.register_forward_hook(forward_hook2)
            hooks.append(hook)
        
        compare_datasets(device, resnet_model)

    else:
        batch_size = 200
        mandatory_dataloader = get_mandatory_dataset(batch_size)
        
        hooks = []
        for module in selected_modules:
            hook = module.register_forward_hook(forward_hook1)    
            hooks.append(hook)

        # Iterate over your dataloader and compute the average non-positive percentage for each feature map
        inputs, _ = next(iter(mandatory_dataloader))
        outputs = model(inputs.to(device))

    # Remove the forward hooks
    for hook in hooks:
        hook.remove()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def compare_datasets(device, model):
    batch_size = 200
    mandatory_dataloader = get_mandatory_dataset(batch_size)
    print("madatory done")
    cifar_dataloader = get_cifar_data(batch_size)
    print("cifar done")
    imagenet_dataloader = get_imagenet_data(batch_size)
    print("imagenet done")

    datasets = [mandatory_dataloader, imagenet_dataloader, cifar_dataloader]
    for dataloader in datasets:
        inputs, _ = next(iter(dataloader))
        upscaled_input = interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=True)
        upscaled_input = upscaled_input.float()
        outputs = model(upscaled_input.to(device))

        plt.show()
            

def get_mandatory_dataset(batch_size):
    train_loader, _, _ = process_data("data", batch_size, testdata=True)
    return train_loader

def get_cifar_data(batch_size):
    data_batch_1 = unpickle("data/cifar-10-batches-py/test_batch")
    data = data_batch_1['data']
    labels = data_batch_1['labels']

    cifar_data = np.empty((batch_size, 3, 32, 32))
    cifar_labels = np.zeros((batch_size))

    for i, img in enumerate(data):
        if i == batch_size:
            break
        img = np.reshape(img, (3, 32, 32))  
        img = (img / 255.0)
        cifar_data[i] = img
        cifar_labels[i] = labels[i]
    
    cifar_data = cifar_data.astype(np.double)
    dl = Image_DL(cifar_data, cifar_labels)

    cifar_dl = DataLoader(dl, batch_size=batch_size)
    return cifar_dl

def get_imagenet_data(batch_size):
    images = []
    labels = []
    count = 0
    for img in os.listdir("data/imagenet/images"):
        if img.endswith('.JPEG'):
            image = Image.open(os.path.join('data/imagenet/images',img))
            image = image.resize((224, 224))
            image = np.array(image)
            if image.ndim == 3 and count < batch_size:
                count += 1
                image = image.astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))
                images.append(image)
                labels.append(0)

    images = np.asarray(images)
    labels = np.asarray(labels)

    train_dl = Image_DL(images, labels)

    train_loader = DataLoader(train_dl, batch_size=batch_size)

    return train_loader

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
    # Reshape the mean feature map tensor to (batch_size, num_channels), assuming num_channels is the last dimension
    mean_feature_map = spatial_mean.view(spatial_mean.size(0), -1)
    covariance_matrix = torch.cov(mean_feature_map)

    # Compute the top-k eigenvalues and sort them in decreasing order
    eigenvalues, _ = torch.linalg.eigh(covariance_matrix, UPLO='U')
    top_k_eigenvalues, _ = torch.sort(eigenvalues, descending=True)[:k]
    plt.bar(range(len(top_k_eigenvalues)), top_k_eigenvalues)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue magnitude')
    plt.title('Top-k eigenvalues of empirical covariance matrix')
    plt.figure()

