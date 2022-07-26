import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import Model

# training
def train(dataloader, net, loss, optim):
    size = len(dataloader.dataset)
    net.train()
    curve = []
    for i, (x,y) in enumerate(dataloader):
        # predict output, calculate loss
        x_p = net(x)
        err = loss(x_p,y)
        # optimize network
        optim.zero_grad()
        err.backward()
        optim.step()
        # display error
        if i%100 == 0:
            print('iter:',i,',loss:',err.item())
            curve.append(err.item())
    return curve

# testing
def test(dataloader, net, loss):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    net.eval()
    
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, (x,y) in enumerate(dataloader):
            x_p = net(x)
            test_loss += loss(x_p, y).item()
            correct += (x_p.argmax(1)==y).sum().item()
            if i%100 == 0:
              print('testing:',str(i)+'/'+str(num_batch))
    test_loss /= num_batch
    correct /= size
    print('test loss:',test_loss,',accuracy:',correct*100)
        


# hyper-parameters
batch_size = 8
num_epoch = 1

# mnist dataset and loader
data_train = datasets.MNIST('./', train=True, download=True, transform=ToTensor())
data_test  = datasets.MNIST('./', train=False, download=True, transform=ToTensor())
dataloader_train = DataLoader(data_train, batch_size=batch_size)
dataloader_test = DataLoader(data_test, batch_size=batch_size)

# model
net = Model()
print(net)

# loss and optim
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=1e-4)

# train network
for t in range(num_epoch):
    print('epoch num:',t)
    curve = train(dataloader_train, net, loss, optim)
    print('testing phase:')
    test(dataloader_test, net, loss)


torch.save(net.state_dict(), 'model.pth')
print('complete!')

f = plt.figure(0)
plt.plot(curve)
plt.show()

