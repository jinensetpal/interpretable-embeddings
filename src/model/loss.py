#!/usr/bin/env python3

from torch import nn
import torch

from src import const


# KLD adapted from https://mbernste.github.io/posts/vae/
class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()
        self.alpha = const.KLD_ALPHA

    def forward(self, recon, X):
        recon, mean, std_dev = recon
        var = std_dev.pow(2)
        return self.mse(recon, X) - self.alpha * torch.sum(1 + torch.log(var) - mean.pow(2) - var)
