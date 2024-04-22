#!/usr/bin/env python3

import torch
import pickle
from src.model.arch import Model
from .base import BaseEncoder
from src import const
from sklearn.decomposition import IncrementalPCA
import os

PATH = os.path.join(os.path.dirname(__file__), '../../../pca.pickle')

class PCA(BaseEncoder):
    def __init__(self, model_name=const.MODEL_NAME):
        #model = unpickle in new pca.py
        with open(PATH, 'rb') as f:
            pca = pickle.load(f)
        self.model = Model().to(const.DEVICE)
        self.model = IncrementalPCA(n_components=768, batch_size=768)
        self.model = pca

    def encode(self, batch):
        #batch wise transform sing pca
        return torch.tensor(self.model.transform(batch.detach().cpu())).to(const.DEVICE).to(torch.float)

    def top_components(self, n=10):
        return self.model.components_[:n]

    def top_features(self, n=10):
        return self.model.explained_variance_ratio_[:n]
