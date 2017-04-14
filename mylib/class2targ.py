# -*- coding=UTF-8 -*-

"""

File name : class2targ.py

Creation date : 02-02-2016

Last modified : Wed 03 Feb 2016 09:46:24 AM BRST

Created by :

Purpose : 

    When making classifiers, the class might come as a categorical variable.
    This must be translated to a neural network output, i.e., if there are 5
    possible classes, the output layer will have 5 neurons. Class 1 will be
    translated to [1, 0, 0, 0, 0] and so on.

Usage :

    targ = class2targ(class_vector), where class_vector is the correct label.
    targ is the correct label translated to a neural network encoding.

Observations :

    class_vector must be integer only. Every class must have at least one
    instance. class_vector should not have 0.

"""

import numpy as np

def class2targ(class_vector):

    # Finding out the number of output neurons.
    no_output = len(set(class_vector))

    targ = np.zeros((class_vector.shape[0], no_output))

    for i in range(targ.shape[0]):

        targ[i, class_vector[i]-1] = 1

    return targ
    





