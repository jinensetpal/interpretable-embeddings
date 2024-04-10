import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

PATH = os.path.join(os.path.dirname(__file__), '../../../pca.pickle')
with open(PATH, 'rb') as f:
  pca = pickle.load(f)

#get top 10 feature names
# number of components
n_pcs= pca.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = pd.read_csv(r"D:\interpretable-embeddings\data\expression\634da99d771a5b05d92a59cf\features_RNA_5prime_data.csv", header=None).iloc[:,0].tolist()
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df = pd.DataFrame(dic.items())
print(df[:10])
