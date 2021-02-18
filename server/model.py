import torch
import torchvision
import timm

import numpy as np

from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class ClassificationModel(nn.Module):
    def __init__(self, arch="resnext101_32x8d", n_classes=6, pretrained=True):
        super().__init__()
        self.model = timm.create_model(arch, pretrained=pretrained)
        input_features = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class DetectionModel(nn.Module):
    def __init__(self, arch="fasterrcnn_resnet50_fpn", n_classes=6, pretrained=True):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            min_size=1024
        )
        input_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            input_features, n_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x
