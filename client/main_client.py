import socket
import Client
import torch
import torchvision

import numpy as np
import train
import Model as m
import data_loader as dl
import transforms as t

from time import time, sleep
from torchvision import datasets, transforms
from torch import nn, optim
from copy import deepcopy
from torch.utils.data import Subset


# HOST = '127.0.0.1'
# PORT = 9865

# for e in range(epochs):
#     running_loss = 0
#     for images, labels in train_loader:
#         # Flatten MNIST images into a 784 long vector
#         images = images.view(images.shape[0], -1)

#         # Training pass
#         optimizer.zero_grad()

#         output = model(images)
#         loss = criterion(output, labels)

#         # This is where the model learns by backpropagating
#         loss.backward()

#         # And optimizes its weights here
#         optimizer.step()

#         running_loss += loss.item()
#     # else:
#     print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
#     temp1 = deepcopy(model)
#     # print(sd1)
#     # SEND TO SERVER
#     torch.save(model.state_dict(), client.model_folder +
#                '/' + client.model_name)
#     client.connect()
#     client.send('UPDATE_MODEL')

#     # RECEIVE FROM SERVER
#     client.connect()
#     result = client.get('MODEL')
#     while result == False:
#         print('\nSERVER HAS NOT UPDATED GLOBAL')
#         sleep(5)
#         client.connect()
#         result = client.get('MODEL')

#     model.load_state_dict(torch.load(
#         client.model_folder + '/' + client.model_name))
#     # print(model.state_dict())
#     for p1, p2 in zip(temp1.parameters(), model.parameters()):
#         if p1.data.ne(p2.data).sum() > 0:
#             print('\nMODELS NOT SAME')
#     print('\nMODELS SAME')


# print("\nTraining Time (in minutes) =", (time()-time0)/60)

# # EVALUATION
# correct_count, all_count = 0, 0
# for images, labels in val_loader:
#     for i in range(len(labels)):
#         img = images[i].view(1, 784)
#         with torch.no_grad():
#             logps = model(img)

#         ps = torch.exp(logps)
#         probab = list(ps.numpy()[0])
#         pred_label = probab.index(max(probab))
#         true_label = labels.numpy()[i]
#         if(true_label == pred_label):
#             correct_count += 1
#         all_count += 1

# print("Number Of Images Tested =", all_count)
# print("\nModel Accuracy =", (correct_count/all_count))


def main(): 
    epochs = 15
    ds_type = "classification"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)


    time0 = time()

    if ds_type == "classification": 
        train_loader, test_loader = dl.get_classification_dataset(
            train_csv_file="",
            train_path="", 
            train_transform=t.transform_train,
            train_labels=True,
            train_bs=32,
            test_csv_file="",
            test_pat="",
            test_transform=t.transform_test,
            test_labels=False,
            test_bs=1
        )

    for epoch in range(epochs):
    
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        running_loss = 0
        # TRAIN 
        running_loss += train.train(model, train_loader, optimizer, criterion, epochs, device)

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




if __name__ == "__main__":
    main()