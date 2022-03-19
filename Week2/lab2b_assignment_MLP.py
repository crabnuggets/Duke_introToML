from random import shuffle
from turtle import forward
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import datasets, transforms
from tqdm import tqdm

class MNIST_MLP(nn.Module): 
    def __init__(self):
        super().__init__()
        # Linear transformation of data size of 783 (images) to a layer of 500 hidden units
        self.fc1 = nn.Linear(784, 500)
        # Linear transformation of second layer of 500 units to desired output dimension of 10 units
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x): 
        # Rectified linear unit for first forward pass to include non-linearities
        z = F.relu(self.fc1(x))
        # Rectified linear unit for second forward pass to include non-linearities
        y = F.relu(self.fc2(z))
        return y

# Load the data
mnist_train = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## Training 
# Instantiate model
myModel = MNIST_MLP()

# Loss and Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(myModel.parameters(), lr=0.1)
trainBatches = iter(train_loader)

# Iterate through train set minibatches
print("Iterating through train set minibatches...")
for batch in tqdm(trainBatches): 
    images, labels = batch
    optimizer.zero_grad()
    x = images.view(-1, 784)
    y = myModel.forward(x)
    loss = criterion(y, labels)
    loss.backward()
    optimizer.step() 

## Testing 
correct = 0
total = len(mnist_test)
testBatches = iter(test_loader)

with torch.no_grad(): 
    ## Iterate through test set minibatches
    print("Iterating through test set minibatches...")
    for batch in tqdm(testBatches):
        images, labels = batch
        x = images.view(-1, 784)
        y = myModel.forward(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))