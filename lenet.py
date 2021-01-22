import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" LeNet5 Pytorch definition """
class LeNet(nn.Module):
    def __init__(self, num_classes, return_activations=False):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.return_activations = return_activations

        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.linear1 = nn.Linear(16*5*5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

        self.activation_length = 6*28*28 + 16*10*10 + 120 + 84 + self.num_classes

    def forward(self, x):
       return self.forward_value(x)

    def forward_value(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2,2))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x
