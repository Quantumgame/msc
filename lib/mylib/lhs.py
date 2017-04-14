# -*- coding=UTF-8 -*-

"""

File name : lhs.py

Creation date : 06-02-2016

Last modified :

Created by :

Purpose :

    Implements Latin Hypercube Sampling.

Usage :

    lhs(10, param_list)

    where 10 is the number of draws, param_list is a list of ml.Parameter
    instances

Observations :

"""

import numpy as np
import mylib as ml

def lhs(n, param_list):

    # Array with all the draws
    draws = np.zeros((len(param_list), n))

    # All arguments in *argv are Parameters
    i = 0
    for arg in param_list:
        assert(isinstance(arg,ml.Parameter) == True), 'Argument is not a \
        Parameter instance'

        draws[i] = arg.sample(n)
        i += 1

    return draws
