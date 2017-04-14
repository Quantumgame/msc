# -*- coding=UTF-8 -*-

"""

File name : build_dict.py

Creation date : 18-11-2016

Last modified :

Created by :

Purpose :
    
    Builds a dictionary connecting the labels (which are integer values) to 
    names. Ex.: 0 -> bottle, 1 -> boot etc.

Usage :

    build_dict.py data_dir save_dir

Observations :

    data_dir and save_dir must be full path

    The data label must be in the last column of the data files

    Data must be in npy format

"""

import os
import sys
import numpy as np

data_dir = sys.argv[1]
save_dir = sys.argv[2]

new_dict = {}
os.chdir(data_dir)
categories = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]


for d in categories:

    os.chdir(d)
    files = [f for f in os.listdir() if f[-3:] == 'npy']
    data = np.load(files[0])
    label = int(data[-1])
    new_dict[d] = label
    print(d)
    os.chdir('..')

np.save(save_dir + '/' + 'dictionary.npy', [new_dict])
