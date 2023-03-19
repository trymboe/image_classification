import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from dataloader import Image_DL
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate

from torchvision.models import resnet18, ResNet18_Weights

def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

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
            hook = module.register_forward_hook(forward_hook2)
        else:
            hook = module.register_forward_hook(forward_hook1)    
        hooks.append(hook)
  
    
    if task3:
        resnet_model = resnet18(weights = ResNet18_Weights.DEFAULT)
        data_batch_1 = unpickle("data/cifar-10-batches-py/test_batch")
        data = data_batch_1['data']
        labels = data_batch_1['labels']

        cifar_data = np.empty((1000, 3, 32, 32))
        cifar_labels = np.zeros((1000))

        for i, img in enumerate(data):
            if i == 1000:
                break
            img = np.reshape(img, (3, 32, 32))  
            img = (img / 255.0)
            cifar_data[i] = img
            cifar_labels[i] = labels[i]
        
        cifar_data = cifar_data.astype(np.double)
        dl = Image_DL(cifar_data, cifar_labels)

        cifar_dl = DataLoader(dl, batch_size=64, drop_last=True)

        datasets = [dataloader,cifar_dl]
        resnet_model.float()

        for dataloader in datasets:
            count = 0
            for inputs, targets in dataloader:
            
                upscaled_input = interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=True)
                outputs = resnet_model(upscaled_input.to(device))
                count += 1
                if count == 200:
                    break
        plt.show()

    else:
        # Iterate over your dataloader and compute the average non-positive percentage for each feature map
        count = 0
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
        dict = pickle.load(fo, encoding='latin1')
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
    plt.figure()

