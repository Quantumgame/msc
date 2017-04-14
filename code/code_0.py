# -*- coding=UTF-8 -*-

"""

File name : code.py

Creation date : 28-04-2016

Last modified :

Created by :

Purpose :

Usage :

Observations :

"""

import vrep
import cv2
import numpy as np
import time
import sys
import mylib as ml
import bottleneck as bn
from saliency import saliency
from scipy.spatial.distance import pdist, squareform

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

def img2rgbnp(image, resolution):
    """
    Transforms image from vrep camera to rgb np so opencv can display it
    properly.

    Receives: image and resolution (tuple).
    Returns: np array
    """

    img = image
    img = np.array(img, dtype=np.uint8)
    img.resize([resolution[0], resolution[1], 3])
    temp = np.copy(img[:,:,2])
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 0] = temp
    img = np.flipud(img)

    return img

def get_x_joint_pos(x_ampl, x_res, x_img_coord, last_x_joint_pos):
    """
    Receives:
    x_ampl: camera x amplitude in rads
    x_res: camera x resolution
    x_img_coord: the desired x coordinate

    Returns:
    x_joint_pos = joint position, in rads, that makes x_img_coord the center of
    the visual field
    """
    
    # transposing the img coord to -127 +128 range (for 256 resolution)
    x_img_coord_ = x_img_coord - x_res/2

    x_joint_pos = -1*(x_ampl * x_img_coord_) / x_res + last_x_joint_pos
    #x_joint_pos = -2*(x_ampl * x_img_coord_) / x_res
    return x_joint_pos


def get_y_joint_pos(y_ampl, y_res, y_img_coord, last_y_joint_pos):
    """
    Receives:
    y_ampl: camera y amplitude in rads
    y_res: camera y resolution
    y_img_coord: the desired y coordinate

    Returns:
    y_joint_pos = joint position, in rads, that makes y_img_coord the center of
    the visual field
    """
    
    # transposing the img coord to -127 +128 range (for 256 resolution)
    y_img_coord_ = -y_img_coord + y_res/2

    y_joint_pos = 1*(y_ampl * y_img_coord_) / y_res + last_y_joint_pos
    #y_joint_pos = 2*(y_ampl * y_img_coord_) / y_res
    return y_joint_pos

def get_n_max_sal(sal, n):

    """
    Receives:
        saliency map
        n, number of points to return

    Returns:
        numpy array with the indices of maximum points
    """

    # max number of candidate indices
    max_cand = n

    # distance threshold for distance based cutting out
    #thresh = 10

    sal_ = sal
    #max_indices = bn.argpartsort(-1*sal_.flatten(), n)[:n]
    size = sal_.shape[0]*sal.shape[1]
    #cand_indices = bn.argpartsort(-1*sal_.flatten(), max_cand)[:max_cand]
    max_indices = bn.argpartsort(-1*sal_.flatten(), n)[:n]

    # pdist needs matrix, not 1d array
    #max_indices = np.reshape(max_indices, (max_indices.shape, 1))

    # iterating over max indices to see if there are points too close to each
    # other. if there is, we must eliminate one of them and pick the next one
    # after the n max saliency points

    """
    incr = 0

    for i in range(n):

        # obtaining distance matrix  
        dist_mat = squareform(pdist(max_indices, 'minkowski', 1)) 

        for j in range(n):

            if (dist_mat[i, j] < thres) and (i != j):

                max_indices[j] = cand_indices[n+incr]
                incr += 1
    """
    max_indices = np.unravel_index(max_indices, sal_.shape)
    print(max_indices)
    return max_indices



########################################################################

#### let the code begin #########

########################################################################

# parameters

# number of most salient points to start segmentation
n = 6


# close all opened connections

vrep.simxFinish(-1)

# start new connection.
# IMPORTANT: there must be the following line in a THREADED child script in vrep:
# simExtRemoteApiStart(19999)

clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 100)

if clientID != -1:
    print('Connected to the remote API server')
else:
    print('Connection not successful.')
    sys.exit('Could not connect.')

# grabbing objects handles

errorCode, camera = vrep.simxGetObjectHandle(clientID,\
        'camera', vrep.simx_opmode_oneshot_wait)

errorCode, joint_z = vrep.simxGetObjectHandle(clientID,\
        'joint_z', vrep.simx_opmode_oneshot_wait)

errorCode, joint_x = vrep.simxGetObjectHandle(clientID,\
        'joint_x', vrep.simx_opmode_oneshot_wait)

#print(errorCode)


# initializing states

moving = 0
waiting_input = 0
#point = (128, 126)

last_joint_pos = (0, 0)

if clientID!=-1:
    res,v0=vrep.simxGetObjectHandle(clientID,'camera',vrep.simx_opmode_oneshot_wait)

    res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
    while (vrep.simxGetConnectionId(clientID)!=-1):
        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)
        if res==vrep.simx_return_ok:

            # converting vrep img to numpy array bgr for proper opencv handling
            img = img2rgbnp(image, resolution)
            cv2.imshow('felipe', img)
            cv2.waitKey(50)

            # will calculate saliency only if not moving and not waiting for
            # input

            if (moving == 0) and (waiting_input == 0):

                #calculating saliency
                sal = get_saliency(img)

                # finding the most salient point
                #max_sal_point = np.unravel_index(np.argmax(sal), sal.shape)

                # finding the N most salient points

                # saliency that can be altered for max sal calculation
                #sal_ = sal
                
                max_indices = get_n_max_sal(sal, n)
                max_sal_point = (max_indices[0][0], max_indices[1][0])

                #cv2.imshow('static sal', sal)
                #cv2.waitKey(500)

                # initializing segmentation
                segm_total = np.zeros((resolution[0], resolution[1], 3))

                # iterating over all salient points
                for i in range(n):

                    # x coordinate is the second element (not sure why...)
                    segm_current = ml.stochastic_reg_grow(img, (max_indices[1][i],\
                        max_indices[0][i]),\
                        15, 15, 10, 30, 20)
                    
                    # transforming segm_current into binary mask
                    segm_current[np.where(np.mean(segm_current, axis=2) != 0)] = 1
                    segm_current = np.asarray(segm_current, dtype=np.bool)
                    
                    # transforming segm_total into binary mask
                    segm_total[np.where(np.mean(segm_total, axis=2) != 0)] = 1
                    segm_total = np.asarray(segm_total, dtype=np.bool)

                    # updating segm_total
                    segm_total = np.bitwise_or(segm_current, segm_total)

                segm_total = np.multiply(segm_total, img)
                cv2.imshow('segm', segm_total)
                cv2.waitKey(50)

                # activate moving state
                moving = 0

            # if moving state is active, move joints, and dont calculate saliency
            
            if moving == 1:

                # move joints

                x_joint_pos = get_x_joint_pos(np.pi/4, 256, max_sal_point[1],\
                        last_joint_pos[0])

                y_joint_pos = get_y_joint_pos(np.pi/4, 256, max_sal_point[0],\
                        last_joint_pos[1])

                vrep.simxSetJointTargetPosition(clientID, joint_z,\
                        x_joint_pos,\
                        vrep.simx_opmode_oneshot)

                vrep.simxSetJointTargetPosition(clientID, joint_x,\
                        y_joint_pos,\
                        vrep.simx_opmode_oneshot)

                last_joint_pos = (x_joint_pos, y_joint_pos)

                cv2.waitKey(500)
                moving = 0
                waiting_input = 0

            cv2.waitKey(500)


    vrep.simxFinish(clientID)



vrep.simxFinish(clientID)

