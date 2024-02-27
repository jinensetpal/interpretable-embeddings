#!/usr/bin/env python3

import scanpy as sc
import random
import torch

from src import const


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split='all'):
        random.seed(const.SEED)
        self.data = sc.read_mtx(const.FEATURES_DIR / 'RNA_5prime_data.mtx').to_df().T

        if split in const.SPLITS:
            self.data['split'] = random.choices(const.SPLITS, weights=const.SPLIT_WEIGHTS, k=len(self.data))
            self.data = self.data[self.data['split'] == split]
            self.data.pop('split')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].tolist())
