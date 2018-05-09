"""
Making your own classifier with self notes
https://www.youtube.com/watch?v=AoeEHqVSNOw

outputs accuracy at the end
"""

import random

# calculates euclidean distance between points
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)


# makes your own neighbor classifier
class eKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        # find the closest neighbor points
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range (1, len(self.x_train)):
            dist = euc (row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

# import iris dataset
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

# split data to test and train 
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y , test_size = .5)

# define classifier
my_classifier = eKNN()

# train classifier
my_classifier.fit(x_train, y_train)

# make predictions
predictions  = my_classifier.predict(x_test)

# print accuracy
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)


