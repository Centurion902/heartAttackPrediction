from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

import pickle
##read data
import numpy as np

file = open("data/splitData", 'rb')
splitData = pickle.load(file)
trainingSetData = splitData['trainingSetData']
trainingSetTarget = splitData['trainingSetTarget']

##train model
clf = svm.SVC(C= 1000, gamma='auto', kernel= 'rbf')
clf.fit(trainingSetData, trainingSetTarget)

##save
fileName = "firstModel"
file = open(fileName, 'wb')
pickle.dump(clf, file)

print("trained and saved to " + fileName)