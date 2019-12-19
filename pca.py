import numpy as np
from sklearn.decomposition import PCA
f = open("data/heartAttackClean.csv")
f.readline()  # skip the header
data = np.loadtxt(f, delimiter=",")
pca = PCA(n_components=data.shape[1])
pca.fit(data)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)
