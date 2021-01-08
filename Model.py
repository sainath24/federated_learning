import torch
import torchvision
import numpy as np
from torch import nn, optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_size = 784
        self.hidden_sizes = [128, 64]
        self.output_size = 10
        self.conv = nn.Sequential(nn.Linear(self.input_size, self.hidden_sizes[0]),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_sizes[1], self.output_size),
                                  nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.conv(x)

    
