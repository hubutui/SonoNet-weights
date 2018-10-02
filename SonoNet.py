#!/bin/env python3
#
# Construct SonoNet according to models.py
# Note that lasage uses different default args from pytorch
# so we need to ajust args here, like bias = False if conv layer is followed by a batchnorm layer
# and eps = 1e-4 for batchnorm layer
import torch.nn as nn


def SonoNet64(in_channels=1, num_classes=14):
    model = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(256, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512, eps=1e-4),
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 256, kernel_size=1, bias=False),
        nn.BatchNorm2d(256, eps=1e-4),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, num_classes, kernel_size=1, bias=False),
        nn.BatchNorm2d(num_classes, eps=1e-4)
    )

    return model

# Also note that Global average pool layer is not added yet
# we could apply
# x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
# or
# x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
# where x is the output of model
