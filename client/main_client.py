import socket
import Client
import torch
import torchvision

import numpy as np
import Model as m

from time import time, sleep
from torchvision import datasets, transforms
from torch import nn, optim
from copy import deepcopy
from torch.utils.data import Subset



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

dataset = torchvision.datasets.MNIST(
    './data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('./data/', download=True,
                        train=False, transform=transform)
indices = np.arange(len(dataset))
train_dataset = Subset(dataset, indices[:len(indices)//2])
train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, num_workers=0, batch_size=64)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# Validation
train_targets = []
for _, target in train_loader:
    train_targets.append(target)
train_targets = torch.cat(train_targets)

print(train_targets.unique(return_counts=True))

dataiter = iter(train_loader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

# HOST = '127.0.0.1'
# PORT = 9865

client = Client.Client()
client.connect()
print('\nCONNECTED TO SERVER')
client.get('MODEL')
print('\nSAVED GLOBAL MODEL')

# GET MODEL
model = m.Model()

model.load_state_dict(torch.load(
    client.model_folder + '/' + client.model_name))

criterion = nn.NLLLoss()
images, labels = next(iter(train_loader))
# images = images.view(images.shape[0], -1)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    # else:
    print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
    temp1 = deepcopy(model)
    # print(sd1)
    # SEND TO SERVER
    torch.save(model.state_dict(), client.model_folder +
               '/' + client.model_name)
    client.connect()
    client.send('UPDATE_MODEL')

    # RECEIVE FROM SERVER
    client.connect()
    result = client.get('MODEL')
    while result == False:
        print('\nSERVER HAS NOT UPDATED GLOBAL')
        sleep(5)
        client.connect()
        result = client.get('MODEL')

    model.load_state_dict(torch.load(
        client.model_folder + '/' + client.model_name))
    # print(model.state_dict())
    for p1, p2 in zip(temp1.parameters(), model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            print('\nMODELS NOT SAME')
    print('\nMODELS SAME')


print("\nTraining Time (in minutes) =", (time()-time0)/60)

# EVALUATION
correct_count, all_count = 0, 0
for images, labels in val_loader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
# TRAINING
