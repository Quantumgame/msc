# -*- coding=UTF-8 -*-

"""

File name : category_search.py

Creation date : 29-07-2016

Last modified :

Created by :

Purpose :

    Performs a category search. Given a category and an object image, returns 
    whether the object is of the given category

Usage :

Observations :

"""

import os
import numpy as np
import sys

imgs_dir = sys.argv[1]

category_dict = {0: 'cone', 1: 'clock', 2: 'bottle', 3: 'laptop', 4: 'boot'}

sgng = np.load('sgng.npy')[0]

os.chdir(imgs_dir)

for f in os.listdir():

    if f[-5:] == 'ascii':
        curr_classeme = np.loadtxt(f)
        curr_bmu, curr_sbmu, bla = sgng._bmu(curr_classeme, 1)
        print(category_dict[curr_bmu.label], category_dict[curr_sbmu.label], f)



