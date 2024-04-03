#!/usr/bin/env python3

from src.data import Dataset
from src import const

import umap.umap_ as umap
import pickle
import umap

if __name__ == '__main__':
    scaler = pickle.load(open((const.MODEL_DIR / 'scaler.pkl'), 'rb'))

    encoder = umap.UMAP(n_components=const.HIDDEN_SIZE, n_neighbors=const.UMAP_NEIGHBORS, random_state=const.SEED)
    encoder.fit(scaler.transform(Dataset('train').data))
    pickle.dump(encoder, open(const.MODEL_DIR / 'umap.pkl', 'wb'))
