from sklearn.model_selection import train_test_split
import pickle
##read data
import numpy as np
import pandas as pd
from sklearn import preprocessing

f = open("data/heartAttackClean.csv")
f.readline()  # skip the header
data = np.loadtxt(f, delimiter=",")


X = data[:, :-1]  # select all but last column
y = data[:, -1]   # select last column (target)

print(X)

enc = preprocessing.OneHotEncoder()
enc.fit(X)
onehotlabels = enc.transform(X).toarray()

trainingSetData, testSetData, trainingSetTarget, testSetTarget = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=22)

print(trainingSetData.shape)
print(trainingSetTarget.shape)

splitData = {
    "trainingSetData": trainingSetData, 
    "testSetData": testSetData, 
    "trainingSetTarget": trainingSetTarget, 
    "testSetTarget":testSetTarget
}
##saving test sets
pickle.dump(splitData, open("splitData", 'wb'))
