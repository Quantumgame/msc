# -*- coding=UTF-8 -*-

"""

File name : generate_true.py

Creation date : 07-10-2016

Last modified :

Created by :

Purpose :

    Generates .true file for Hans LARFDSSOM pipeline.
    The .true file is saved in the same location of the original file.

Usage :

    generate_true.py file.arff

Observations :

"""

import mylib as ml
import numpy as np

def generate_true(arff_file):

    content = ml.read_arff(arff_file)

    labels = np.array(list(set(content[:, -1])))

    true_filename = arff_file + '.true'
    true_file = open(true_filename, 'w')

    dims = content.shape[1] - 1
    true_file.write('DIM=%d;FILE=%s\n' %(dims, arff_file))

    count = np.zeros(labels.shape[0])
    indices = []

    for i in range(labels.shape[0]):

        indices.append([])

        for j in range(dims):

            true_file.write('1 ')

        for k in range(content.shape[0]):

            if content[k, -1] == labels[i]:

                count[i] += 1
                indices[i].append(k)

        true_file.write('%d ' %(count[i]))
        true_file.write('%d' %(indices[i][0]))
        for m in range(1, len(indices[i])):
            true_file.write(' %d' %(indices[i][m]))

        true_file.write('\n')

    true_file.close()
