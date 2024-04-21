#!/usr/bin/env python3

from src import const
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = const.MODEL_NAME

        unit_scope = [const.N_FEATURES // 2 ** pw for pw in range(const.DEPTH)]
        self.downsamplers = nn.ModuleList([nn.LazyLinear(units) for units in unit_scope[1:]])
        self.upsamplers = nn.ModuleList([nn.LazyLinear(units) for units in unit_scope[::-1]])
        self.relu = nn.ReLU()

        self.encoded = nn.LazyLinear(const.HIDDEN_SIZE)

    def forward(self, x):
        for downsampler in self.downsamplers:
            x = downsampler(x)
            x = self.relu(x)
        enc = self.encoded(x)

        if self.training:
            x = enc
            for upsampler in self.upsamplers:
                x = upsampler(x)
                x = self.relu(x)
            return x, enc
        else: return enc


if __name__ == '__main__':
    model = Model().to(const.DEVICE)
    model.eval()
