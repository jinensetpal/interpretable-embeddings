#!/usr/bin/env python3

import torch

from src.model.arch import Model
from .base import BaseEncoder
from src import const


class AutoEncoder(BaseEncoder):
    def __init__(self, model_name=const.MODEL_NAME):
        self.model = Model().to(const.DEVICE)
        self.model.load_state_dict(torch.load(const.MODEL_DIR / f'{model_name}.pt',
                                              map_location=const.DEVICE))
        self.model.eval()

    def encode(self, batch):
        with torch.no_grad():
            return self.model(batch.to(const.DEVICE))[1]
