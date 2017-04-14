# -*- coding=UTF-8 -*-

"""

File name : flatten.py

Creation date : 21-02-2016

Last modified :

Created by :

Purpose :

    Flattens a list of lists, possibly irregular ones.

Usage :

    ml.flatten(list_of_lists)

Observations :

    Flattens only one level. Ex.:

    [[0, 1], [1, 2, 3]] will become
    [0, 1, 1, 2, 3]

    But [[[0, 1], [1, 2]], [[2, 4], [2, 1]]] will become
    [[0, 1, 1, 2], [2, 4, 2, 1]]

"""

def flatten(list_of_list):

    flat = [y for x in list_of_list for y in x]
    return flat
