import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets

# model definition
class Model(nn.Module):

    def __init__(self):
        super(type(self), self).__init__()
        # conv and fc objects
        self.conv1 = nn.Conv2d(1,8,3,2)
        self.conv2 = nn.Conv2d(8,16,3,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# hyper-parameters
batch_size = 4
num_epoch = 2


# mnist dataset
data_train = datasets.MNIST('./', train=True, download=True)
data_test  = datasets.MNIST('./', train=False, download=True)
# model
net = Model()

x = torch.rand(8,1,32,32)

y = net(x)

print(y.shape)

	
		

