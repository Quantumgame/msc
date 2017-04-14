# -*- coding=UTF-8 -*-

"""

File name : svm2npy.py

Creation date : 17-10-2016

Last modified :

Created by :

Purpose :

    Converts file from libsvm format to npy.

Usage :

    svm2npy.py filename

Observations :

"""

import numpy as np
import sys
import sklearn.datasets as sk

filename = sys.argv[1]

data, label = sk.load_svmlight_file(filename)

new_array = np.zeros((data.shape[0], data.shape[1]))

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        new_array[i, j] = data[i, j]
    if label[i] == -1:
        label[i] = 0

label = np.reshape(label, (label.shape[0], 1))
final = np.hstack((new_array, label))
np.save(filename, final)
