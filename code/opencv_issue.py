# -*- coding=UTF-8 -*-

"""

File name : opencv_issue.py

Creation date : 10-08-2016

Last modified :

Created by :

Purpose :

Usage :

Observations :

"""

import numpy as np
import cv2
import os

img_list = ['teste11.png', 'teste12.png', 'teste13.png', 'teste9.png']

os.chdir('imgs/test')
for img in img_list:
    curr_img = cv2.imread(img)
    cv2.imshow('hello', curr_img)
    cv2.waitKey(500)
