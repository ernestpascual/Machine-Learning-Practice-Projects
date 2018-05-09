"""
Based from this tutorial:
https://www.youtube.com/watch?v=cKxRvEZd3Mw

Just added something to convert 1s and 0s into oranges and apples
Added notes
"""
from sklearn import tree

# STEP 1: Collect Training Data 
# Note that apples are described as 0, oranges are 1
features = [[140,1], [130,1], [150,1], [170,0]]
labels = [0,0,1,1]

# STEP 2: Train a classifier - Decision Tree
# Make an empty classifier
clf = tree.DecisionTreeClassifier()

# fit: create rules, training algorithm, find patterns and data
clf.fit(features, labels)

# test a new fruit, predict if apples or oranges
if clf.predict([[20,1]]) == 1:
    print 'orange'
else:
    print 'apple'

