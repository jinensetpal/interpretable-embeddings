print("0")
from src.data import Dataset
from src import const
import pickle
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

print("a")
if __name__ == '__main__':

    df = Dataset("train").data
    print("b")

    #std = df.std(0)
    #mean = df.mean(0)

    print("b2")
    
    scaler = StandardScaler()
    #.fit(df)

    scaler.scale_ = 0.3838406501392982 
    scaler.mean_ = 0.07388965334090265 
    print("b3")

    normalized_data = scaler.transform(df)
    #normalized_data = scaler.fit_transform(df)
    del df

    print("c")

    #pca = PCA(n_components=768)
    #pca.fit(normalized_data)
    ipca = IncrementalPCA(n_components=const.HIDDEN_SIZE, batch_size=768)
    ipca.fit(normalized_data)

    with open('pca.pickle', 'wb') as f:
        pickle.dump(ipca, f)
