#!/usr/bin/env python3

import scanpy as sc
import torch

from src import const


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = sc.read_mtx(const.FEATURES_DIR / 'RNA_5prime_data.mtx').to_df().T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].tolist())
