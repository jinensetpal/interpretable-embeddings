#!/usr/bin/env python3

import torch
import pickle
from src.model.arch import Model
from .base import BaseEncoder
from src import const


class PCA(BaseEncoder):
    def __init__(self, model_name=const.MODEL_NAME):
        #model = unpickle in new pca.py
        with open('pca.pickle', 'rb') as f:
            pca = pickle.load(f)
        self.model = pca
        # self.model = Model().to(const.DEVICE)
        # self.model.load_state_dict(torch.load(const.MODEL_DIR / f'{model_name}.pt',
        #                                       map_location=const.DEVICE))
        # self.model.eval()

    def encode(self, batch):
        with torch.no_grad():
            return self.model(batch)[1]
