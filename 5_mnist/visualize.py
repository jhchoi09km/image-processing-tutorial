import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import Model

# unit test and visualization
batch_size = 8

# model
net = Model()
net.load_state_dict(torch.load('model.pth'))
print('model load complete')

# db
data_test = datasets.MNIST('./', train=False, download=True, transform=ToTensor())
dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)
dataloader_iter = iter(dataloader_test)

# get data
img_total = []
for b in range(batch_size):
    x,y = next(dataloader_iter)
    x_p = net(x)
    correct = (x_p.argmax(1)==y).type(torch.int)

    img = []
    for i in range(len(x)):
        x_r = torch.repeat_interleave(x[i],3,0)
        x_r[2,-2:,:] = 0

        if correct[i] == 1:
            x_r[0,-2:,:], x_r[1,-2:,:] = 0,1
        else:
            x_r[0,-2:,:], x_r[1,-2:,:] = 1,0

        img.append(x_r)
         
    img = torch.cat(img, dim=2).permute(1,2,0)
    img_total.append(img)

img_total = torch.cat(img_total, dim=0)
plt.imshow(img_total.numpy())
plt.show()

