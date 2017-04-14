# -*- coding=UTF-8 -*-

"""

File name : parameter_analysis_sgng.py

Creation date : 30-06-2016

Last modified :

Created by :

Purpose :

    Runs a parameter analysis on sgng. Receives the data filename, the number
    of runs and the parameters range.

Usage :

Observations :

    SGNG save file is sgng.npy by default.

"""

import sys
import mylib as ml
import numpy as np
import categorize_images_sgng as cat
import os

def parse(args, online_bulk):

    """
    Receives args and looks for known parameters.
    Returns list of ml.Parameters.
    """

    i = 0
    if online_bulk == 'bulk':

        for i in range(len(args)):

            if args[i] == 'winner_lr':
                # Instantiating a paramter with low = next argument; high = next
                # next arg
                winner_lr = ml.Parameter('winner_lr', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'neigh_lr':
                neigh_lr = ml.Parameter('neigh_lr', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'error_decay':
                error_decay = ml.Parameter('error_decay', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'max_age':
                max_age = ml.Parameter('max_age', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'max_epoch':
                max_epoch = ml.Parameter('max_epoch', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'epoch_ins':
                epoch_ins = ml.Parameter('epoch_ins', float(args[i+1]), float(args[i+2]))

        return [winner_lr, neigh_lr, error_decay, max_age, max_epoch, epoch_ins]

    elif online_bulk == 'online':

        for i in range(len(args)):

            if args[i] == 'winner_lr':
                # Instantiating a paramter with low = next argument; high = next
                # next arg
                winner_lr = ml.Parameter('winner_lr', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'neigh_lr':
                neigh_lr = ml.Parameter('neigh_lr', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'error_decay':
                error_decay = ml.Parameter('error_decay', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'max_age':
                max_age = ml.Parameter('max_age', float(args[i+1]), float(args[i+2]))
            elif args[i] == 'max_epoch':
                max_epoch = ml.Parameter('max_epoch', float(args[i+1]), float(args[i+2]))

        return [winner_lr, neigh_lr, error_decay, max_age, max_epoch]

if __name__ == '__main__':

    filename = sys.argv[1]
    n = int(sys.argv[2])
    qty_train = float(sys.argv[3])
    online_bulk = sys.argv[4]
    save_dir = sys.argv[5]
    param_list = parse(sys.argv, online_bulk)


    choice = 'caltech'

    if choice == 'caltech':
        data_full = np.load(filename)
        data = data_full[:, 0:-1]
        # Last element of npy files is label
        label = data_full[:, -1]
    else:
        data, label = ml.read_dataset(filename, classification=False, norm=True)
    """
    elif choice == 101:
        data_full = ml.read_arff(filename)
        data = data_full[:, 0:-1]
        label = data_full[:, -1]
    """
        

    #data = data_full[:, 0:-1]
    #label = data_full[:, -1]

    values = ml.lhs(n, param_list)

    print('accuracy std', end=' ')
    if online_bulk == 'online':
        print('ocmr std', end=' ')

    ml.print_header(param_list)

    if save_dir == 'n':
        save_boolean = 0
    else:
        save_boolean = 1
        os.chdir(save_dir)


    for i in range(n):

        winner_lr_c = values[0, i] 
        neigh_lr_c = values[1, i] 
        error_decay_c = values[2, i] 
        max_age_c = int(values[3, i]) 
        max_epoch_c = int(values[4, i]) 

        if online_bulk == "bulk":

            epoch_ins_c = int(values[5, i]) 

            cat.run_bulk(data, label, qty_train, winner_lr_c, neigh_lr_c,\
                    error_decay_c, max_age_c, max_epoch_c, epoch_ins_c, save_boolean)

        elif online_bulk == "online":

            cat.run_online(data, label, qty_train, winner_lr_c,\
                    neigh_lr_c, error_decay_c, max_age_c, max_epoch_c, save_boolean)

