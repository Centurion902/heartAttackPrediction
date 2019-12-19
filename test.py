from sklearn import datasets, metrics, svm
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import pickle

file = open("data/splitData", 'rb')
splitData = pickle.load(file)
testSetData = splitData['testSetData']
testSetTarget = splitData['testSetTarget']

file = open("firstModel", 'rb')
clf = pickle.load(file)

##confusion matrix in terminal
disp = plot_confusion_matrix(clf, testSetData, testSetTarget)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

##fancy graphic confusion matrix
class_names = ["0", "1"]
np.set_printoptions(precision=2)


disp = plot_confusion_matrix(clf, testSetData, testSetTarget,
                                display_labels=class_names,
                                cmap=plt.cm.Blues,
                                normalize='true')
disp.figure_.suptitle("Normalized confusion matrix")
print("Normalized Confusion matrix:\n%s" %disp.confusion_matrix)

print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(testSetTarget, clf.predict(testSetData))))

plt.show()