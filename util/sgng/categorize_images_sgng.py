# -*- coding=UTF-8 -*-

"""

File name : categorize_images.py

Creation date : 30-06-2016

Last modified :

Created by :

Purpose :

    Runs a round of image categorization. Receives an arff file with data +
    label (last column) and sgng parameters.

Usage :

Observations :

"""

import numpy as np
import mylib as ml
import SGNG
import sys
import numpy.matlib as matlib
import os


def run_bulk(data, label, qty_train=0.7, winner_lr=0.05, neigh_lr=0.0006,\
        error_decay=0.2, max_age=50, max_epoch=1000,\
        epoch_ins=50, save_boolean=1):

    runs = 5

    acc_vec = np.zeros(runs)
    for i in range(runs):

        sgng = SGNG.SGNG(winner_lr = winner_lr, neigh_lr = neigh_lr, error_decay =\
                error_decay, max_age=max_age)

        sgng.train_bulk(data, label, qty_train, max_epoch, epoch_ins)

        test_data = sgng.test_set[:, 0:-1]
        test_label = sgng.test_set[:, -1]
        acc = sgng.test_bulk(test_data, test_label)

        acc_vec[i] = acc

    acc_final = np.mean(acc_vec)
    std_final = np.std(acc_vec)
    print("%.3f %.3f %.3f %.5f %.3f %d %d %d" %(acc_final, std_final, winner_lr, neigh_lr,\
        error_decay, max_age, max_epoch, epoch_ins), flush=True)

    if save_boolean == 1:
        sgng_files = [i for i in os.listdir() if 'sgng' in i]
        sgng.write_to_file('sgng' + str(len(sgng_files)))


def run_online(data, label, qty_train=0.7, winner_lr=0.05,\
        neigh_lr=0.0006, error_decay=0.2, max_age=50, max_epoch=1000, save_boolean=1):

    runs = 10
    acc_vec = np.zeros(runs)
    error_count = np.zeros(runs)

    for i in range(runs):

        sgng = SGNG.SGNG(winner_lr=winner_lr, neigh_lr=neigh_lr,\
                error_decay=error_decay, max_age=max_age)

        train, test = sgng._split_train_test(data, label, qty_train,\
                perm=True)

        train = np.matlib.repmat(train, max_epoch, 1)
        np.random.shuffle(train)

        train_data = train[:, 0:-1]
        train_label = train[:, -1]
        test_data = test[:, 0:-1]
        test_label = test[:, -1]

        for j in range(train_data.shape[0]):

            sgng.train_single(train_data[j], train_label[j])

        acc = sgng.test_bulk(test_data, test_label)

        acc_vec[i] = acc
        error_count[i] = sgng.error_count / train_data.shape[0]

        #sgng = None

    acc_final = np.mean(acc_vec)
    std_final = np.std(acc_vec)
    error_count_final = np.mean(error_count)
    error_count_std = np.std(error_count)
    print("%.3f %.3f %.3f %.3f %.3f %.5f %.3f %d %d" %(acc_final, std_final, error_count_final, error_count_std, winner_lr, neigh_lr,\
        error_decay, max_age, max_epoch), flush=True)

    #if save_boolean == 1:
    #    sgng.write_to_file('sgng')

    if save_boolean == 1:
        sgng_files = [i for i in os.listdir() if 'sgng' in i]
        sgng.write_to_file('sgng' + str(len(sgng_files)))

if __name__ == '__main__':

    """
    if len(sys.argv) != 6:
        print("Usage: categorize_images filename train_percent max_age\
 max_epoch epoch_ins")
    """

    filename = sys.argv[1]
    winner_lr = float(sys.argv[2])
    neigh_lr = float(sys.argv[3])
    error_decay = float(sys.argv[4])
    train_percent = float(sys.argv[5])
    max_age = int(sys.argv[6])
    max_epoch = int(sys.argv[7])
    epoch_ins = int(sys.argv[8])
    online_bulk = sys.argv[9]

    full_data = ml.read_arff(filename)

    data = full_data[:, 0:-1]
    label = full_data[:, -1]

    if online_bulk == 'bulk':

        run_bulk(data, label, train_percent, winner_lr, neigh_lr, error_decay,\
                max_age, max_epoch, epoch_ins)

    elif online_bulk == 'online':

        run_online(data, label, train_percent, winner_lr, neigh_lr, error_decay,\
                max_age, max_epoch)




