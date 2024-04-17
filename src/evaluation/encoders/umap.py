#!/usr/bin/env python3

import pickle
import torch

from .base import BaseEncoder
from src import const


class UMAPEncoder(BaseEncoder):
    def __init__(self, model_name=const.MODEL_NAME):
        self.scaler = pickle.load(open((const.MODEL_DIR / 'scaler.pkl'), 'rb'))
        self.umap = pickle.load(open((const.MODEL_DIR / 'umap.pkl'), 'rb'))

    def encode(self, batch):
        with torch.no_grad():
            return torch.tensor(self.umap.transform(self.scaler.transform(batch.detach().cpu()))).to(const.DEVICE)
