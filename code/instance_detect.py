# -*- coding=UTF-8 -*-

"""

File name : instance_detect.py

Creation date : 01-08-2016

Last modified :

Created by :

Purpose :

    Given a model directory with several views of a desired object,
    the query image classeme file and the models directory with several other objects, returns 
    the model whose classeme is the closest to the query classeme.

Usage :

Observations :

"""

import numpy as np
import os
import sys
from saliency import saliency

def get_saliency(img):
    """
    Calculates img saliency, using NVT algorithm (Itti, 1998)
    """

    intensty = saliency.intensityConspicuity(img)
    gabor = saliency.gaborConspicuity(img, 4)

    im = saliency.makeNormalizedColorChannels(img)
    rg = saliency.rgConspicuity(im)
    by = saliency.byConspicuity(im)
    c = rg + by
    sal = 1./3 * (saliency.N(intensty) + saliency.N(c) + saliency.N(gabor))
    sal = cv2.resize(sal, dsize=(img.shape[1],img.shape[0])) 
    #sal = (sal + magno)/2
    sal = sal.astype(np.uint8)

    return sal

def distance(v1, v2):

    # Returns the euclidean distance between v1 and v2

    assert v1.shape == v2.shape, "V1 and V2 must be of same shape"

    return np.sqrt(np.sum(np.square(v1 - v2)))


def instance_detect(desired_obj, q_classeme, models_dir, thresh):

    first_dir = os.getcwd()
    os.chdir(models_dir + '/' + desired_obj)

    for curr_pose in os.listdir():

        if curr_pose[-5:] == 'ascii':

            curr_classeme = np.loadtxt(curr_pose)

            if np.abs(distance(curr_classeme, q_classeme)) <= thresh:

                os.chdir(first_dir)
                return 1

    os.chdir(first_dir)
    return 0



"""
def instance_detect(desired_obj, q_classeme, models_dir, thresh):

    #desired_obj = sys.argv[1]
    #q_classeme = np.loadtxt(sys.argv[2])
    #models_dir = sys.argv[3]

    first_dir = os.getcwd()
    os.chdir(models_dir)

    # Initializing dist and output detection string (first and second choices)
    smallest_dist = np.infty
    smallest_dist2 = np.infty
    output_detection = None
    output_detection2 = None

    assert desired_obj in os.listdir(), 'Desired object not in models directory'

    for curr_dir in os.listdir():

        if os.path.isdir(curr_dir):

            os.chdir(curr_dir)
            for curr_file in os.listdir():
            
                if curr_file[-5:] == 'ascii':

                    curr_classeme = np.loadtxt(curr_file)
                    curr_dist = distance(curr_classeme, q_classeme) 
                    if curr_dist < smallest_dist:
                        temp = smallest_dist
                        smallest_dist = curr_dist
                        if curr_dir != output_detection:
                            output_detection2 = output_detection
                            smallest_dist2 = temp
                        output_detection = curr_dir

            os.chdir('..')

    #print(output_detection, output_detection2, smallest_dist, smallest_dist2)
    # if sd and sd2 are too far away from each other, 
    # disregard output2

    os.chdir(first_dir)
    if smallest_dist < thresh * smallest_dist2:

        output_detection2 = output_detection

    if desired_obj in (output_detection, output_detection2):

        return 1
    
    else:

        return 0

"""

def run(desired_obj, imgs_dir, models_dir, thresh):

    imgs_files = os.listdir(imgs_dir)

    for curr_file in imgs_files:

        if curr_file[-5:] == 'ascii':

            #print(os.getcwd())
            curr_classeme = np.loadtxt(imgs_dir + '/' + curr_file)

            if instance_detect(desired_obj, curr_classeme, models_dir, thresh):

                print(curr_file)

if __name__ == '__main__':

    desired_obj = sys.argv[1]
    imgs_dir = sys.argv[2]
    models_dir = sys.argv[3]
    thresh = float(sys.argv[4])

    run(desired_obj, imgs_dir, models_dir, thresh)

    #print(instance_detect(desired_obj, q_classeme, models_dir, thresh))

