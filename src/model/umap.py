#!/usr/bin/env python3

from src.data import Dataset

import umap.umap_ as umap
import pickle
import umap

if __name__ == '__main__':
    scaler = pickle.load(open((const.MODEL_DIR / 'scaler.pkl'), 'rb'))
    df = scaler.transform(Dataset('train').data)

    reducer = umap.UMAP(n_components=const.HIDDEN_SIZE, n_neighbors=const.UMAP_NEIGHBORS, random_state=const.SEED)
    pickle.dump(open(const.MODELS_DIR / 'umap.pkl', 'wb'))
