#!/usr/bin/env python3

from pathlib import Path
import random
import torch

# paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
FEATURES_DIR = DATA_DIR / 'expression' / '634da99d771a5b05d92a59cf'

# data
STRATIFY_AGAINST = ['disease', 'library_preparation_protocol', 'sex']
SPLITS = ['train', 'valid', 'test']
SPLIT_WEIGHTS = [0.7, 0.05, 0.25]
N_FEATURES = 29066
SEED = 10

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EARLY_STOPPING_THRESHOLD = 5
LR_BOUNDS = [1E-4, 1E-1]
MODEL_NAME = 'default'
LEARNING_RATE = 1E-3
UMAP_NEIGHBORS = 40
HIDDEN_SIZE = 768
BATCH_SIZE = 1024
MIN_EPOCHS = 20
KLD_ALPHA = 1E-2
ONLINE = False
EPOCHS = 50
DEPTH = 5

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/interpretable-embeddings.mlflow'
LOG_REMOTE = False

random.seed(SEED)
