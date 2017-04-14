# -*- coding=UTF-8 -*-

"""

File name : analyze_model.py

Creation date : 02-08-2016

Last modified :

Created by :

Purpose :

    Analyzes a model, taking snapshots of the object in different 
    poses.

Usage :

Observations :

"""

import vrep
import os
import sys
import cv2
import numpy as np


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

def analyze_model(model_path_and_name, qty_poses=9, save_location='imgs/models/'):

    """
    Takes several qty_poses 'pictures' of a VREP model in different 
    orientations. Saves each of the pictures in save_location.

    Each of the picture will be SIFT-described for instance detection.

    clientID is the VREP scene ID.

    qty_poses must be divisible by 3
    """

    assert qty_poses % 3 == 0, 'qty_poses must be divisible by 3' 

    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 100)

    if clientID != -1:
        print('Connected to the remote API server')
    else:
        print('Connection not successful.')
        sys.exit('Could not connect.')



    # finding model name
    model_name = ''
    for i in model_path_and_name:
        if i == '/':
            model_name = ''
        elif i == '.':
            break
        else:
            model_name += i

    # creating models directory, if not existent
    if model_name not in os.listdir(save_location):
        os.mkdir(save_location + '/' + model_name)

    # loading model and positioning
    res, model_handle = vrep.simxLoadModel(clientID, modelPathAndName=model_path_and_name, options=1, operationMode=vrep.simx_opmode_oneshot_wait)
    res = vrep.simxSetObjectPosition(clientID, model_handle, -1, (-0.9, 0, 5), vrep.simx_opmode_oneshot)

    count = 0

    # grabbing camera2 image, saving it and rotating model
    if clientID!=-1:

        res,v1=vrep.simxGetObjectHandle(clientID,'camera2',vrep.simx_opmode_oneshot_wait)
        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v1,0,vrep.simx_opmode_streaming)

        while (vrep.simxGetConnectionId(clientID)!=-1) and count < qty_poses:

            cv2.waitKey(200)
            res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v1,0,vrep.simx_opmode_buffer)
            if res==vrep.simx_return_ok:
    
                #cv2.waitKey(200)
                #img = img2rgbnp(image, resolution)
                #cv2.imshow('foo', img)
                #cv2.waitKey(50)

                # Cases for count: 
                # when count < qty_poses / 3, poses are obtained rotating x axis 
                # when count >= qty_poses / 3 and < qty_poses*2/3, poses are obtained rotating y axis 
                # else, poses are obtained rotating z axis

                
                
                if count < qty_poses / 3:

                    cv2.waitKey(200)
                    res = vrep.simxSetObjectOrientation(clientID, model_handle,\
                            #model_handle, (360*3/qty_poses, 0, 0), vrep.simx_opmode_oneshot)
                            model_handle, (2*np.pi*3/qty_poses, 0, 0), vrep.simx_opmode_oneshot)
                    cv2.waitKey(200)

                elif count < (qty_poses*2/3):

                    cv2.waitKey(200)
                    res = vrep.simxSetObjectOrientation(clientID, model_handle,\
                            model_handle, (0, 2*np.pi*3/qty_poses, 0), vrep.simx_opmode_oneshot)
                    cv2.waitKey(200)

                else:

                    cv2.waitKey(200)
                    res = vrep.simxSetObjectOrientation(clientID, model_handle,\
                            model_handle, (0, 0, 2*np.pi*3/qty_poses), vrep.simx_opmode_oneshot)
                    cv2.waitKey(200)
                
                """
                cv2.waitKey(200)
                res = vrep.simxSetObjectOrientation(clientID, model_handle,\
                        model_handle, (0, 2*np.pi/qty_poses, 0), vrep.simx_opmode_oneshot)
                cv2.waitKey(200)
                """
                #else:

                #    break

                cv2.waitKey(200)
                res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v1,0,vrep.simx_opmode_buffer)
                img = img2rgbnp(image, resolution)
                cv2.imwrite(save_location +'/' + model_name + '/' + model_name + '_' + str(count) + '.png', img)
                count += 1


    cv2.waitKey(200)
    ret = vrep.simxRemoveModel(clientID, model_handle, vrep.simx_opmode_oneshot)
    cv2.waitKey(200)
    #ret = vrep.simxRemoveModel(clientID, model_handle, vrep.simx_opmode_oneshot)
    print(ret)

if __name__ == '__main__':

    model = sys.argv[1]
    poses = int(sys.argv[2])
    analyze_model(model, poses)
    
