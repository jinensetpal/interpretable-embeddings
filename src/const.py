#!/usr/bin/env python3

from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
FEATURES_DIR = DATA_DIR / 'expression' / '634da99d771a5b05d92a59cf'

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'default'
LEARNING_RATE = 1E-3
LR_BOUNDS = [1E-4, 1E-1]
N_FEATURES = 29066
HIDDEN_SIZE = 768
BATCH_SIZE = 1024
EPOCHS = 50
DEPTH = 5

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/interpretable-embeddings.mlflow'
LOG_REMOTE = True
