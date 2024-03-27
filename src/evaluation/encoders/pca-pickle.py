from ...data import Dataset
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = Dataset("train").data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df)

pca = PCA(n_components=768)
pca.fit(normalized_data)

with open('pca.pickle', 'wb') as f:
    pickle.dump(pca, f)
