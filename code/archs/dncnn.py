from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os, glob, datetime, time
import re
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim


class DnCNN(nn.Module):
    """
    This is a modified implementation of the DnCNN from https://github.com/cszn/DnCNN
    """
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        self.image_channels = image_channels
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        lastcnn = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lastcnn = m
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        init.constant_(lastcnn.weight, 0)


if __name__ == '__main__':
    from IPython import embed
    embed()
    model = DnCNN()
