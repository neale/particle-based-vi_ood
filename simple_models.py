import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 1)

    def forward(self, x):
       return self.forward_value(x)

    def forward_value(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 2)

    def forward(self, x):
       return self.forward_value(x)

    def forward_value(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

