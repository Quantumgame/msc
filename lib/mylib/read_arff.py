# -*- coding=UTF-8 -*-

"""

File name : read_arff.py

Creation date : 29-06-2016

Last modified :

Created by :

Purpose :

    Reads arff file. Basically looks for the first line after @data. All data
    after @data are stored in a matrix. Each file line goes to a matrix row.

Usage :

    data = read_arff(filename)

Observations :

"""

import mylib as ml
import numpy as np

def read_arff(filename):

    f = open(filename, 'r')

    string = f.readline()

    while string != '@data\n':

        string = f.readline()

    data = [line.split('\t') for line in f]

    for i in range(len(data)):

        data[i][-1] = data[i][-1].strip()
    
    return np.asarray(data, np.float)
