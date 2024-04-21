#!/usr/bin/env python3

import torch

from src.model.vae import Model as VAE
from src.model.ae import Model as AE
from .base import BaseEncoder
from src import const


class AutoEncoder(BaseEncoder):
    def __init__(self, model_name=const.MODEL_NAME, train=False):
        if model_name.startswith('vae'): self.model = VAE().to(const.DEVICE)
        else: self.model = AE().to(const.DEVICE)

        if not train:
            self.model.load_state_dict(torch.load(const.MODEL_DIR / f'{model_name}.pt',
                                                  map_location=const.DEVICE))
            self.model.eval()
        self.grad = train

    def encode(self, batch):
        if self.grad: return self.model(batch.to(const.DEVICE))
        with torch.no_grad():
            return self.model(batch.to(const.DEVICE))
