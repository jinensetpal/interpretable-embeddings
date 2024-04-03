#!/usr/bin/env python3

import torch

from .base import BaseEncoder
from src import const


class UMAP(BaseEncoder):
    def __init__(self, model_name=const.MODEL_NAME):
        self.scaler = pickle.load(open((const.MODEL_DIR / 'scaler.pkl'), 'rb'))

        self.encoder = umap.UMAP(n_components=const.HIDDEN_SIZE, n_neighbors=const.UMAP_NEIGHBORS, random_state=const.SEED)

    def encode(self, batch):
        with torch.no_grad():
            return self.model(batch.to(const.DEVICE))[1]
