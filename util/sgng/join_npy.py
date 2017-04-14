# -*- coding=UTF-8 -*-

"""

File name : join_npy.py

Creation date : 01-07-2016

Last modified :

Created by :

Purpose :

    Joins all npy files into a single data file, data.npy

Usage :

    join_npy.py 5categ_256

Observations :

"""

import os
import numpy as np
import sys

data_dir = sys.argv[1]

os.chdir(data_dir)

files = os.popen('ls', 'r')

first_file = files.readline().strip()

data = []

if first_file[-4:] != '.npy':

    first_file = files.readline().strip()

#data = np.load(first_file)
data.append(np.load(first_file))

while 1:

    curr_file = files.readline().strip()

    if not curr_file:
        break

    if curr_file[-4:] == '.npy':

        array = np.load(curr_file)

        #data = np.vstack((data, array))
        data.append(array)

data = np.array(data)
np.save('data.npy', data)


