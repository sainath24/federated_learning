import sys
sys.path.append("../")

import socket
import Client
import torch
import torchvision

import numpy as np
import train
import model as m
import data_loader as dl
import transforms as t

from time import time, sleep
from torchvision import datasets, transforms
from torch import nn, optim
from copy import deepcopy
from torch.utils.data import Subset
labels = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

train_loader, test_loader = dl.get_classification_dataset(
            train_csv_file="D:\\fyp_data\\train_folds_0_to_3.csv.gz",
            train_path="D:\\fyp_data\\proc", 
            train_transform=t.transform_train,
            train_labels=labels,
            train_bs=32,
            test_csv_file="D:\\fyp_data\\val_fold_4.csv.gz",
            test_path="D:\\fyp_data\\proc",
            test_transform=t.transform_test,
            test_labels=[],
            test_bs=1
        )

# Plot train example

batch = next(iter(train_loader))
print(batch["image"][0])