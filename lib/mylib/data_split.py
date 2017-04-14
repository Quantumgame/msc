# -*- coding=UTF-8 -*-

"""

File name : data_split.py

Creation date : 02-02-2016

Last modified : Wed 03 Feb 2016 09:47:09 AM BRST

Created by :

Purpose :

    Splits data into training and testing bunches. Data instances are chosen
    randomly.

Usage :

    train_data, test_data = data_split(data, target, r),
    where r is the proportion of training data.

    Example:

    train_data, train_target, test_data, test_target = data_split(data, target, 0.75)

Observations :

"""

import numpy as np
import mylib as ml

def data_split(data, target, r):

    size = data.shape[0]
    targ_dim = target.shape[1]
    data_dim = data.shape[1]

    full_data = np.hstack((data, target))

    np.random.shuffle(full_data)
    train_data = full_data[0:round(r*size), 0:data_dim]
    train_target = full_data[0:round(r*size), data_dim:]
    
    test_data = full_data[round(r*size):, 0:data_dim]
    test_target = full_data[round(r*size):, data_dim:]

    return train_data, train_target, test_data, test_target



