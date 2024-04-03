#!/usr/bin/env python3

from src import const

from torch.nn import functional as F
from torch import nn
import torch

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        return F.binary_cross_entropy(y_pred, y) + KL
