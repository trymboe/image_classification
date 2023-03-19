import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torchvision import models

WIDTH = 150
HEIGHT = 150

CLASSES_LIST = ["forest", "buildings",
            "glacier", "street",
            "mountain", "sea"]

class ResNet18(nn.Module):
    def __init__(self, nn_inputs, h, w, outputs, device, weight_init="kaiminghe" ) -> None:
        super().__init__()

        self.nn_inputs = nn_inputs
        self.h = h
        self.w = w
        self.outputs = outputs
        self.weight_init = weight_init

        net = models.resnet18()

        original_layer = net.conv1

        net.conv1 = nn.Conv2d(
            self.nn_inputs,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            dilation=original_layer.dilation,
            bias=original_layer.bias,
        )

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, self.outputs)

        self.net = net

    def forward(self, inputs):
        x = self.net(inputs)
        return x
    

def create_model(device, lr, momentum):
    '''
    Creates a ResNet18 PyTorch model for image classification.

    Args:
        device (torch.device): The device on which to create the model.

    Returns:
        - torch.nn.Module: A ResNet18 model.
        - torch.nn.CrossEntropyLoss: The loss function used for training the model.
        - torch.optim.Optimizer: The optimizer used for updating the model weights during training.
    '''
    resnet18 = ResNet18(3, HEIGHT, WIDTH, len(CLASSES_LIST), device).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18.parameters(), lr=lr, momentum=momentum)
    return resnet18, criterion, optimizer