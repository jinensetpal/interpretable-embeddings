#!/usr/bin/env python3

from sklearn.preprocessing import StandardScaler
from itertools import product
import scanpy as sc
import pandas as pd
import random
import pickle
import torch

from src import const


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split='all'):
        random.seed(const.SEED)
        self.data = sc.read_mtx(const.FEATURES_DIR / 'RNA_5prime_data.mtx').to_df().T
        self.metadata = pd.read_csv(const.FEATURES_DIR / 'cells_RNA_5prime_data.csv', header=None)

        self.metadata = pd.merge(self.metadata, pd.read_csv(const.DATA_DIR / 'metadata' / 'combined_metadata.csv')[1:], how='inner', left_on=[0], right_on=['NAME'])

        if split in const.SPLITS:
            for values in product(*[list(self.metadata[column].unique()) for column in const.STRATIFY_AGAINST]):
                subset = self.metadata.eval(' & '.join([f'`{key}`=="{value}"' for key, value in zip(const.STRATIFY_AGAINST, values)]))
                self.metadata.loc[subset, 'split'] = random.choices(const.SPLITS, weights=const.SPLIT_WEIGHTS, k=sum(subset))

            indices = self.metadata.eval(f'`split`=="{split}"').tolist()
            self.data = self.data[indices]
            self.metadata = self.metadata[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].tolist()), torch.tensor(float(self.metadata.iloc[idx]['disease'] == 'MONDO_0018874')).unsqueeze(0)


def save_std_scaler():
    ds = Dataset('train')
    scaler = StandardScaler()
    scaler.fit(ds.data)

    pickle.dump(scaler, open(const.MODEL_DIR / 'scaler.pkl', 'wb'))


if __name__ == '__main__':
    save_std_scaler()
