# -*- coding=UTF-8 -*-

"""

File name : print_header.py

Creation date : 07-02-2016

Last modified :

Created by :

Purpose : 

    Print header to std_out.

Usage :

    print_header(param_list)
    
    where param_list is a list of ml.Parameter 

Observations :

"""

import mylib as ml

def print_header(param_list):

    for arg in param_list:
        print(arg.name, end=" "),

    print()
