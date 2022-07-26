import torch
import torch.nn as nn
import torch.nn.functional as F

# model definition
class Model(nn.Module):

    def __init__(self):
        super(type(self), self).__init__()
        # conv and fc objects
        self.conv1 = nn.Conv2d(1,8,3,2)
        self.conv2 = nn.Conv2d(8,16,3,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(576,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

