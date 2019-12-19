from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, classification_report


import pickle
##read data
import numpy as np

file = open("data/splitData", 'rb')
splitData = pickle.load(file)
trainingSetData = splitData['trainingSetData']
trainingSetTarget = splitData['trainingSetTarget']
testSetData = splitData['testSetData']
testSetTarget = splitData['testSetTarget']

svm = SVC(kernel="poly", gamma='scale')

# Set up possible values of parameters to optimize over
tuned_parameters = [
    {
        "C": [1, 10, 100, 1000, 10000],
        "degree": [1,2,3,4,5,6,7,8,9],
        'kernel': ['poly'],
    },
    {
        "C": [1, 10, 100, 1000],
        'kernel': ['rbf'],
        'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale', 'auto']
    }
]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(trainingSetData, trainingSetTarget)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print(classification_report(testSetTarget, clf.predict(testSetData)))

##plt.show()