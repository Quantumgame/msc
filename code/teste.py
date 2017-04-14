# -*- coding=UTF-8 -*-

"""

File name : teste.py

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
from saliency import saliency

def get_x_joint_pos(x_ampl, x_res, x_img_coord):
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
    x_img_coord = x_img_coord - x_res/2

    x_joint_pos = (x_ampl * x_img_coord) / x_res
    return x_joint_pos


def get_y_joint_pos(y_ampl, y_res, y_img_coord):
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
    y_img_coord = y_img_coord - y_res/2

    y_joint_pos = (y_ampl * y_img_coord) / y_res
    return -y_joint_pos


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

print(errorCode)

i = 0
max_sal_arg = [21, 40]

if clientID!=-1:
    res,v0=vrep.simxGetObjectHandle(clientID,'camera',vrep.simx_opmode_oneshot_wait)

    res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
    while (vrep.simxGetConnectionId(clientID)!=-1):
        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)
        if res==vrep.simx_return_ok:

            # converting image to rgb np array
            img = image
            img = np.array(img, dtype=np.uint8)
            img.resize([resolution[0], resolution[1], 3])
            temp = np.copy(img[:,:,2])
            img[:, :, 2] = img[:, :, 0]
            img[:, :, 0] = temp
            img = np.flipud(img)
            cv2.imshow('felipe', img)
            cv2.waitKey(2)

            #calculating saliency
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

            # finding the most salient point
            #max_sal_arg = np.unravel_index(np.argmax(sal), sal.shape)
            #print(max_sal_arg)

            cv2.imshow('static sal', sal)
            cv2.waitKey(2)

            # move joints
            vrep.simxSetJointTargetPosition(clientID, joint_z,\
                    get_x_joint_pos(np.pi/4, 256, max_sal_arg[0]),\
                    vrep.simx_opmode_oneshot)

            vrep.simxSetJointTargetPosition(clientID, joint_x,\
                    get_y_joint_pos(np.pi/4, 256, max_sal_arg[1]),\
                    vrep.simx_opmode_oneshot)

            # segmentation
            segm = ml.stochastic_reg_grow(img, (105, 110), 15, 15, 6,\
                    30, 20)
            cv2.imshow('segm', segm)
            cv2.waitKey(10)


    vrep.simxFinish(clientID)



vrep.simxFinish(clientID)

