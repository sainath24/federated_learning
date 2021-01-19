import socket
import torch
import torchvision
import yaml

import train
import Client

import numpy as np

import model as m
import data_loader as dl
import transforms as t

from time import time, sleep
from torchvision import datasets, transforms
from torch import nn, optim
from copy import deepcopy
from torch.utils.data import Subset

def check_model_similarity(model1, model2): 
    """
    Check if two models are the same.
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            print('\nMODELS NOT SAME')
        print('\nMODELS SAME')

def main():
    with open('client_config.yaml') as file:
        config = yaml.safe_load(file)
    epochs = config['epochs']

    lr = config['lr']
    ds_type = config['dataset']
    if config['device']:
        device = torch.device(config['device'])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/10243, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/10243, 1), 'GB')
    
    labels = config['labels']
    n_classes = config['n_classes']
    assert len(labels)==n_classes, "ERR: Number of classes should match labels."

    client = Client.Client(config)
    client.connect()
    print('\nCONNECTED TO SERVER')
    client.get()
    print('\nSAVED GLOBAL MODEL')

    # GET MODEL
    model = m.Model(config["arch"], n_classes)

    model.load_state_dict(torch.load(
        client.model_folder + '/' + client.model_name))

    criterion = torch.nn.BCEWithLogitsLoss()
    optim_params = [{'params': model.parameters(), 'lr': lr}]
    if config["optimizer"]=="SGD": 
        optimizer = optim.SGD(optim_params)
    elif config["optimizer"]=="Adam":
        optimizer = optim.Adam(optim_params)
    else: 
        # default
        optimizer = optim.Adam(optim_params)

    time0 = time()

    if ds_type == "classification":
        train_loader, test_loader = dl.get_classification_dataset(
            train_csv_file=config["train_csv_file"],
            train_path=config["train_path"],
            train_transform=t.transform_train,
            train_labels=labels,
            train_bs=config["train_batch_size"],
            test_csv_file=config["test_csv_file"],
            test_path=config["test_path"],
            test_transform=t.transform_test,
            test_labels=[],
            test_bs=config["test_batch_size"]
        )
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        running_loss = 0
        # TRAIN
        running_loss += train.train(model, train_loader,
                                    optimizer, criterion, epochs, device)

        print("Epoch {} - Training loss: {}".format(epoch,
                                                    running_loss/len(train_loader)))
        temp1 = deepcopy(model)

        # SEND TO SERVER
        torch.save(model.state_dict(), client.model_folder +
                   '/' + client.model_name)
        client.connect()
        client.send()

        # RECEIVE FROM SERVER
        client.connect()
        result = client.get()
        while result == False:
            print('\nSERVER HAS NOT UPDATED GLOBAL')
            sleep(5)
            client.connect()
            result = client.get()

        model.load_state_dict(torch.load(
            client.model_folder + '/' + client.model_name))
        check_model_similarity(temp1, model)

    print("\nTraining Time (in minutes) =", (time()-time0)/60)


if __name__ == "__main__":
    main()
