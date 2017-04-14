# -*- coding=UTF-8 -*-

"""

File name : targ2class.py

Creation date : 12-02-2016

Last modified :

Created by :

Purpose :

    Transforms target-like arrays into class-like arrays.

    Ex.: 0 1 0 0 0 into 2
         0 0 1 0 0 into 3

Usage :

    targ2class(target)

Observations :

    Target lines must have only one 1.
    Target is a matrix

"""

import numpy as np

def targ2class(target):

    class_ = np.zeros(target.shape[0])
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
        
            if target[i, j] == 1:
                class_[i] = j+1
                break

    return class_
