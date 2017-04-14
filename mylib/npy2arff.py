# -*- coding=UTF-8 -*-

"""

File name : npy2arff.py

Creation date : 07-10-2016

Last modified :

Created by :

Purpose :

    Generates an arff file out of a numpy npy file.
    The arff file is saved in the same location of the npy file.

Usage :

    npy2arff filename

Observations :

    Last column must be class attribute.

"""

import numpy as np

def npy2arff(filename):

    arff_filename = filename + '.arff'
    content = np.load(filename)

    arff_file = open(arff_filename, 'w')
    arff_file.write('@relation features\n')

    qty_attributes = content.shape[1] - 1
    for i in range(qty_attributes):
        arff_file.write('@attribute f%d numeric\n' %(i))

    labels = set(np.asarray(content[:, -1], np.int))

    arff_file.write('@attribute class %s\n' %(str(labels)))

    arff_file.write('@data\n')

    for i in range(content.shape[0]):
        arff_file.write('%f' %(content[i, 0]))
        for j in range(1, content.shape[1]):
            if j == content.shape[1]-1:
                arff_file.write('\t%d' %(int(content[i, j])))
            else:
                arff_file.write('\t%f' %(content[i, j]))
        arff_file.write('\n')

    arff_file.close()


if __name__ == "__main__":

    import sys
    npy2arff(sys.argv[1])
