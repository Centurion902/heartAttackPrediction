from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

f = open("data/HCV-Egy-Data.csv")
f.readline()  # skip the header
data = np.loadtxt(f, delimiter=",")

##scaling data
scaler = MinMaxScaler()
print(scaler.fit(data))
data = scaler.transform(data)

from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(data)
print("PCA")
print(pca.explained_variance_) 

clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
clust.fit(data)

## reduce to n most usefull components
pca.n_components = 28
X_reduced = pca.fit_transform(X)

print("PCA Reduced")
print(pca.explained_variance_) 