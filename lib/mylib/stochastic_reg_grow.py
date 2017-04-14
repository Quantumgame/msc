# -*- coding=UTF-8 -*-

"""

File name : stochastic_reg_grow.py

Creation date : 11-05-2016

Last modified :

Created by :

Purpose :

    Given an image, a seed and a threshold, returns a rough segmentation based on
    region similarity.

Usage :

    out_img = stochastic_reg_grow(img, seed, l_thresh, h_thresh, l, n, it)

Observations :

    Here's how it works. First, a filled square of side l around the seed is
    included in the final segmentation. Afterwards, n pixels from the square
    contour are randomly sampled. If the color distance between a sampled pixel
    and the seed is inside the threshold interval, it is included in the
    segmentation. A smaller square is then filled around this sample pixel. And
    it goes on until no more regions satisfy the thresholds.

    l must be even

    seed must be a tuple

"""

import numpy as np
import mylib as ml
import cv2
import scipy as sp
import warnings

def sample(mask, n):

    """
    Sample new seeds from binary mask. n is the number of seeds to be sampled
    """

    # retrieving contour via morphological gradient
    # struct element
    struct_el = np.ones((2, 2), dtype=np.int8)
    contour = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, struct_el)

    # retrieving indices of non-zero elements in contour
    # each element is the pixel coordinate
    nonzero = np.transpose(np.nonzero(contour))

    # if n is too large, clip it to indices length. every pixel in the contour
    # will be a seed
    if n >= len(nonzero):

        seeds = nonzero

    else:

        # sampling n values between 0 and len(nonzero)
        samples = np.random.choice(len(nonzero), n, replace=False)
        
        seeds = nonzero[samples]

    return seeds


def stochastic_reg_grow(img, seed, l_thres, h_thres, l, n, it):

    """
    receives RGB figure and segments it based on region growing
    """

    # filtering annoying warnings
    warnings.filterwarnings("ignore")
    # final segmentation mask
    # must include it*l to account for possible seeds on the edges of the image
    rows = img.shape[0]
    cols = img.shape[1]
    mask = np.zeros((rows+it*l, cols+it*l), dtype=np.uint8)
    seed = tuple(reversed(seed))
    it = int(it)
    seed = (int(seed[0]), int(seed[1]))

    # copying img to the same size as mask to make things easier
    # the 1023 addition is to make the padded part equal to 1023 so the region
    # growing wont invade the padding part
    img_ = np.zeros((rows+it*l, cols+it*l, 3)) + 1023
    img_[(it*l//2):(-it*l//2), (it*l//2):(-it*l//2)] = img
   
    # building the square around the seed
    mask[(int(seed[0]+it*l//2-l//2)):(int(seed[0]+it*l//2+l//2+1)),\
            (int(seed[1]+it*l//2-l//2)):int((seed[1]+it*l//2+l//2+1))] = 1

    #new_seeds = sample(mask[l/2:rows+l/2, l/2:cols+l/2], n)

    # initial quantity of ones in the mask
    # if this number doesnt change after an iteration, stop the process
    ones = 0

    for i in range(0, it):

        # finding new seeds
        # important: seeds are in mask coordinates
        new_seeds = sample(mask, n)

        # iterating over seeds
        for i in range(len(new_seeds)):

            # checking whether the new seeds are similar to the original seed
            # coordinates of new seeds are in mask; coords of seed are in img

            #print(img_[tuple(new_seeds[i])][0])
            #print(img[seed][0])
            print(seed)
            print(new_seeds[i])

            if ((img_[tuple(new_seeds[i])][2] > (img[seed][2] - l_thres)) and \
            (img_[tuple(new_seeds[i])][2] <\
                    (img[seed][2] + h_thres))) and \
            ((img_[tuple(new_seeds[i])][1] > (img[seed][1] - l_thres)) and \
            (img_[tuple(new_seeds[i])][1] <\
                    (img[seed][1] + h_thres))) and \
            ((img_[tuple(new_seeds[i])][0] > (img[seed][0] - l_thres)) and \
            (img_[tuple(new_seeds[i])][0] <\
                    (img[seed][0] + h_thres))):

                # building square around every seed
                mask[int((new_seeds[i, 0]-l/2)):int((new_seeds[i, 0]+l/2+1)),\
                        int((new_seeds[i, 1]-l/2)):int((new_seeds[i, 1]+l/2+1))] = 1

        # number of ones in current mask
        curr_ones = len(np.nonzero(mask)[0])

        if curr_ones == ones:

            break

        else:

            # update ones
            ones = curr_ones

    mask = mask[int((it*l/2)):int((-it*l/2)), int((it*l/2)):int((-it*l/2))]
    mask_ = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask_[np.nonzero(mask)] = 1

    new_img = np.multiply(img, mask_)/255

    return new_img


if __name__ == '__main__':

    #img = np.ones((6,6))

    
    img = np.array([[1, 2, 1, 2, 8, 9, 7, 6, 9],\
                    [0, 1, 2, 1, 6, 7, 5, 4, 9],\
                    [1, 2, 1, 0, 8, 7, 9, 5, 6],\
                    [0, 2, 2, 1, 7, 5, 8, 9, 6],\
                    [2, 1, 2, 1, 8, 7, 8, 9, 0],\
                    [6, 7, 8, 6, 4, 8, 9, 7, 6],\
                    [7, 7, 7, 7, 7, 7, 7, 7, 7],\
                    [8, 7, 6, 7, 8, 7, 6, 7, 8]])

    img_ = np.ones((img.shape[0], img.shape[1], 3))
    img_[:,:,0] = img
    img_[:,:,1] = img
    img_[:,:,2] = img
    img = img_
   

    #img = sp.ndimage.imread('teste2.png', mode='RGB')
    #img = cv2.imread('teste2.png', 1)

    #segment(img, seed, l_thres, h_thresh, l, n, it):
    #def stochastic_reg_grow(img, seed, l_thres, h_thres, l, n, it):
    new_img = stochastic_reg_grow(img, (0, 4), 2, 2, 2, 4, 20)
    #print(new_img)




