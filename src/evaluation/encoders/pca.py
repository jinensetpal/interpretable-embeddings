#!/usr/bin/env python3

import torch
import pickle
from src.model.arch import Model
from .base import BaseEncoder
from src import const
from sklearn.decomposition import PCA, IncrementalPCA
import os

PATH = os.path.join(os.path.dirname(__file__), '../../../pca.pickle')

class PCA(BaseEncoder):
    def __init__(self, model_name=const.MODEL_NAME):
        #model = unpickle in new pca.py
        with open(PATH, 'rb') as f:
            pca = pickle.load(f)
        self.model = pca

    def encode(self, batch):
        #batch wise transform sing pca
        return self.model.transform(batch)
