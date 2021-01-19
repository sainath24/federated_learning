import torch
import torchvision
import timm

import numpy as np

from torch import nn


class Model(nn.Module):
    def __init__(self, arch="resnext101_32x8d", n_classes=6, pretrained=True):
        super().__init__()
        self.model = timm.create_model(arch, pretrained=pretrained)
        input_features = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x
