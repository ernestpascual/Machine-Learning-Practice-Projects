"""
xxp update with addtl notes
"""
from __future__ import print_function

import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels (target_column)
# n_classes = number of classes, in these class 0 - not survived and class 1 - survived
# use load_csv() to load data from csv to python list
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


# Preprocessing function
# Remove fields that are not need in the analysis
def preprocess(passengers, columns_to_delete):
    # Sort by descending id and delete columns
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        [passenger.pop(column_to_delete) for passenger in passengers]
    for i in range(len(passengers)):
        # Converting 'sex' field to float (id is 1 after removing labels column)
        passengers[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(passengers, dtype=np.float64)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
# Added as an arguement for preprocess function specifying which columns to delete
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)

# Build neural network
# 3 layer neural network
# Data input shape is [None, 6] ('None' stands for an unknown dimension, 
# so we can change the total number of samples that are processed in a batch).
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
# model wrapper ('DNN') that automatically performs neural network classifier tasks,
# such as training, prediction, save/restore, and more. 
model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)
# Epochs means how many times the network sees all data with batch_size 
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])