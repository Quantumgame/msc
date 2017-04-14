# -*- coding=UTF-8 -*-

"""

File name : read_dataset.py

Creation date : 02-02-2016

Last modified : Fri 05 Feb 2016 07:44:31 PM BRST

Created by : felipeduque

Purpose :

    Reads a dataset and returns two arrays: the data and the class.
    Unset norm if you dont want normalized data
    Set classification if you're working with classification (i.e., if outputs
    are class labels, instead of desired real values). Setting classification
    will convert integer labels to binary form. Ex., for a 4 class problem:

    1 -> 0001
    2 -> 0010 etc

    Set pointset if you're dealing with pointsets, tipically images. Normally,
    each data row is a different sample. However, image representation is
    tipically a variable-length list of pointsets. Several rows may represent a
    single image. 

    Setting pointset requires an auxiliary .desc file. This is generated
    automatically when calling the binary. See more under help.

Usage :

    read_dataset('wine', norm=True, classification=True, pointset=False)

Observations :

    Datasets available: glass, iris, letter, satellite, vowel, wine, liver

    Other: arff

"""

import numpy as np
import sys
import mylib as ml
import sklearn.preprocessing as sk

def read_dataset(dataset, norm=True, classification=True, pointset=False):

    if dataset == 'wine':

        # wine has classes in the first column; delimiter is comma

        text = np.loadtxt('/home/felipe/mylibs/databases/wine.data', delimiter=',')
        data = text[:,1:]
        categ = text[:,0]

    elif dataset == 'satellite':

        # classes in the last column; delimiter space

        text = np.loadtxt('/home/felipe/mylibs/databases/sat.trn', delimiter=' ')
        data = text[:,:-1]
        categ = text[:,-1]

        # there is no category6, but there is categ 7
        categ[categ==7] -= 1

    elif dataset == 'vowel':

        # classes in the last column; delimiter space.
        # first three columns worthless.
        # has categ with value = 0. 

        text = np.loadtxt('/home/felipe/mylibs/databases/vowel.data')
        data = text[:,3:-1]
        categ = text[:,-1] + 1
        
    elif dataset == 'glass':

        # classes in the last column; delimiter comma
        # first column worthless

        text = np.loadtxt('/home/felipe/mylibs/databases/glass.data', delimiter=',')
        data = text[:,1:-1]
        categ = np.asarray(text[:,-1], dtype=np.int)
        #categ[categ>4] -= 1

    elif dataset == 'iris':

        # classes in the last column; delimiter comma
       
        text = np.loadtxt('/home/felipe/mylibs/databases/DatasetIris.txt', delimiter=',')
        data = text[:,:-1]
        categ = text[:,-1]
        
    elif dataset == 'liver':

        # classes in the last column; delimiter comma
       
        text = np.loadtxt('/home/felipe/mylibs/databases/bupa.data', delimiter=',')
        data = text[:,:-1]
        categ = text[:,-1]

    elif dataset == 'letter':

        # classes in the first column; delimiter comma
        # class is a char. must make it an integer.

        # building dictionary
        dict_ = build_dict()
       
        text = np.loadtxt('/home/felipe/mylibs/databases/letter.data', delimiter=',', converters={0:get})
        #text = np.loadtxt('/home/felipe/mylibs/databases/letter.data', delimiter=',')
        #data = text[:,:-1]
        data = text[:,1:]
        categ = text[:,0]

    elif 'arff' in dataset:

        # classes in the last column.

        text = read_arff(dataset)
        data = text[:,:-1]
        categ = text[:,-1]

    else:

        raise ValueError("Dataset doesn't exist")

    if norm == True:
        data = normalize(data)

    if pointset == True:
        # Will transform data matrix to a pointset list
        data, categ = data2psl(dataset, data, categ)

    if classification == True:
        categ = ml.class2targ(categ)


    return data, categ

def build_dict():
    """
    Specific for the letter database. Builds dictionary of capital letters,
    relating them to integers.
    """

    dict_ = {}

    for i in range(ord('A'), ord('Z')+1):
        
        dict_[chr(i)] = float(i - ord('A') + 1)

    return dict_


def get(letter):

    dict_ = build_dict()
    return dict_[chr(letter[0])]

def data2psl(dataset, data, categ):

    """
    Transforms data matrix to a list of pointset, where each pointset is an
    input sample. Makes use of auxiliary .desc file. 
    """

    aux_filename = dataset + '.desc'
    aux = np.loadtxt(aux_filename)

    # Size of point set list will be: number of rows equal number of elements in aux file (each
    # element of aux file refers to a single point set), number of columns
    # equal number of data columns
    data_psl = []
    categ_psl = np.zeros(aux.shape[0])

    # 'program count' browsing data matrix
    pc = 0
    for i in range(aux.shape[0]):
        data_psl.append(data[pc:(pc+aux[i])])
        categ_psl[i] = categ[pc]
        pc = pc + aux[i]

    return data_psl, categ_psl


def read_arff(filename):

    """
    Reads arff file.
    """

    f = open(filename, 'r')

    # Must make sure all the lines starting with @ are discarded
    # Must read the file and stop when all the @ are done. The line number will
    # be recorded and be passed to skiprows paramter of np.loadtxt

    is_at = True
    i = 0
    while is_at == True:

        if f.readline()[0] == '@':
            i += 1
        else:
            is_at = False

    data = np.loadtxt(f, skiprows=i)
    return data


def normalize(vet):

    """
    vetNorm = np.empty([vet.shape[0], vet.shape[1]])

    for i in range(vet.shape[0]):
        for j in range(vet.shape[1]):
            vetNorm[i, j] = (vet[i, j] - np.min(vet[:,j])) /\
            (np.max(vet[:,j])-np.min(vet[:,j])+0.001) 
    """

    vetNorm = sk.normalize(vet, axis=0)

    # Zeroing vetNorm mean
    vetNorm = vetNorm - np.mean(vetNorm, 0)

    return vetNorm



def letter2int(letter_col):
    """
    Specific for the letter database. Transforms a capital letter into an
    integer. Letter 'A' turns into 1, 'B' to 2 etc.
    """

    numbers = np.zeros(len(letter_col))
    for i in range(len(letter_col)):
        numbers[i] = ord(letter_col[i]) - ord('A') + 1
    return numbers
