# -*- coding=UTF-8 -*-

"""

File name : code_2.py

Creation date : 02-06-2016

Last modified :

Created by :

Purpose :

Usage :

Observations :

    Difference from code.py:

        gets one max saliency point at a time, in order to prevent points too
        close from each other

"""

import vrep
import cv2
import numpy as np
import time
import sys
import mylib as ml
import bottleneck as bn
import os
from saliency import saliency
from scipy.spatial.distance import pdist, squareform

def rectangle_detect(sali):
    """
    recebe img salientada, retorna (L,R,T,B)
    """
    
    mult = 1
    width = sali.shape[1]
    height = sali.shape[0]
    offset_x = 1
    offset_y = 1
    # forcando primeira e ultima linha/coluna sendo 0 pra facilitar
    # o algoritmo

    #sali[0] = 1
    #sali[:,0] = 1
    #sali[-1] = 1
    #sali[:-1] = 1
    
    # detectando L. basta iterar nas colunas
    # pegando a partir do 5 pixel diferente de zero
    count = 0
    for j in range(width):
        if (sali[:,j] >= mult*np.ones((height,3))).any():
            count += 1
            if count == offset_x:
                L = j
                break;

    # detectando R
    count = 0
    for j in reversed(range(width)):
        if (sali[:,j] >= mult*np.ones((height,3))).any():
            count += 1
            if count == offset_x:
                R = j
                break;

    # detectando T 
    count = 0
    for i in range(height):
        if (sali[i] >= mult*np.ones((width,3))).any():
            count += 1
            if count == offset_y:
                T = i
                break;
    # detectando B
    count = 0
    for i in reversed(range(height)):
        if (sali[i] >= mult*np.ones((width,3))).any():
            count += 1
            if count == offset_y:
                B = i
                break;
    return (T, B, L, R)

def transfere(sali, orig):

    """
    recebe imagem salientada, original e retorna imagem redimensionada de
    acordo com a bounding box
    """

    box = rectangle_detect(sali)
    #print(box[0], box[1], box[2], box[3])
    cropped = orig[box[0]:box[1],box[2]:box[3]]
    return cropped



def extract_single_classeme(img):

    os.chdir('imgs')
    if 'temp' not in os.listdir():
        os.mkdir('temp')

    cv2.imwrite('temp/temp.png', img)
    tf = open('listimages.txt', 'w')
    tf.write('temp.png\n')

    # Classeme binary location
    binary_dir = '/mnt/home/felipe/Documents/Faculdade/mestrado/projeto/codigo/classeme/vlg_extractor-master/vlg_extractor'

    # Classemes type
    classemes_type = 'ASCII'

    # Parameters location
    param_dir = '/mnt/home/felipe/Documents/Faculdade/mestrado/projeto/codigo/classeme/vlg_extractor-master/data/'

    # listimages.txt location
    listimages_dir = '/mnt/home/felipe/Documents/Faculdade/mestrado/projeto/codigo/code/imgs/temp/listimages.txt'

    # image location
    #img_dir = '.'
    img_dir = '/mnt/home/felipe/Documents/Faculdade/mestrado/projeto/codigo/code/imgs/temp'

    # where to save the classeme file
    #save_dir = '.'
    save_dir = '/mnt/home/felipe/Documents/Faculdade/mestrado/projeto/codigo/code/imgs/temp'

    command = binary_dir + ' --extract_classemes=' + classemes_type + ' --parameters-dir=' + param_dir + ' ' + listimages_dir + ' ' + img_dir +\
' ' + save_dir

    #os.system(command)
    
    os.chdir('..')
    cv2.waitKey(100)

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

    sal_ = np.copy(sal)

    # number of pixels to make zero in sal_ to prevent picking top salient points too
    # close to each other
    radius = 40

    # max_indices[0][0] is the x coord of the max saliency
    # max_indices[1][0] is the y coord
    max_indices = [np.zeros(n), np.zeros(n)]

    for i in range(n):
        
        # finding the index of sal_ max
        max_curr_ind = np.unravel_index(np.argmax(sal_), sal_.shape)
        max_indices[0][i] = max_curr_ind[0]
        max_indices[1][i] = max_curr_ind[1]

        # making it zero the surroundings of sal_ at max_curr_ind 
        # making sure the zero square doesnt extrapolate sal_ dimensions. if it
        # does, make radius zero
        if ((np.asarray(max_curr_ind) + radius) >= sal_.shape[0:2]).any() or\
             ((np.asarray(max_curr_ind) - radius) <= 0).any(): 
            radius_ = 0
        else:
            radius_ = radius
        
        # filling the square with 0
        if max_curr_ind[0] - radius < 0:
            ind00 = 0
        else:
            ind00 = max_curr_ind[0] - radius

        if max_curr_ind[0] + radius > sal_.shape[0]:
            ind01 = sal_.shape[0]
        else:
            ind01 = max_curr_ind[0] + radius

        if max_curr_ind[1] - radius < 0:
            ind10 = 0
        else:
            ind10 = max_curr_ind[1] - radius

        if max_curr_ind[1] + radius > sal_.shape[1]:
            ind11 = sal_.shape[1]
        else:
            ind11 = max_curr_ind[1] + radius

        #sal_[(max_curr_ind[0] - radius_):(max_curr_ind[0] + radius_),\
        #        (max_curr_ind[1] - radius_):(max_curr_ind[1] + radius_)]\
        #        = 0
        sal_[ind00:ind01, ind10:ind11] = 0
        sal_[max_curr_ind[0], max_curr_ind[1]] = 0

    print(max_indices)

    return max_indices


def analyze_model(clientID, model_path_and_name, qty_poses=9, save_location='imgs/models/'):

    """
    Takes several qty_poses 'pictures' of a VREP model in different 
    orientations. Saves each of the pictures in save_location.

    Each of the picture will be SIFT-described for instance detection.

    clientID is the VREP scene ID.

    qty_poses must be divisible by 3
    """

    assert qty_poses % 3 == 0, 'qty_poses must be divisible by 3' 

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


    ret = vrep.simxRemoveModel(clientID, model_handle, vrep.simx_opmode_oneshot)


                





########################################################################

#### let the code begin #########

########################################################################

# parameters

# number of most salient points to start segmentation
n = 40


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

# analyze

#analyze_model(clientID, '/home/felipe/vrep/chalice/chalice.ttm', 6)


last_joint_pos = (0, 0)

if clientID!=-1:
    res,v0=vrep.simxGetObjectHandle(clientID,'camera',vrep.simx_opmode_oneshot_wait)

    res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
    print(res)
    while (vrep.simxGetConnectionId(clientID)!=-1):
        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)
        if res==vrep.simx_return_ok:

            # converting vrep img to numpy array bgr for proper opencv handling
            img = img2rgbnp(image, resolution)
            cv2.imshow('felipe', img)
            cv2.waitKey(50)
            #cv2.imwrite('imgs/insts/scene.png', img)

            # will calculate saliency only if not moving and not waiting for
            # input

            if (moving == 0) and (waiting_input == 0):

                #calculating saliency
                sal = get_saliency(img)

                # finding the most salient point
                #max_sal_point = np.unravel_index(np.argmax(sal), sal.shape)

                # finding the N most salient points

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
                        30, 30, 5, 30, 40)
                    
                    # transforming segm_current into binary mask
                    segm_current[np.where(np.mean(segm_current, axis=2) != 0)] = 1
                    segm_current = np.asarray(segm_current, dtype=np.bool)
                    
                    # transforming segm_total into binary mask
                    segm_total[np.where(np.mean(segm_total, axis=2) != 0)] = 1
                    segm_total = np.asarray(segm_total, dtype=np.bool)

                    # updating segm_total
                    segm_total = np.bitwise_or(segm_current, segm_total)

                    img_parcial = np.multiply(segm_current, img)

                    # processing current image
                    img_parcial = transfere(img_parcial, img)
                    cv2.imshow('segm', img_parcial)
                    #extract_single_classeme(img_parcial)
                    cv2.imwrite('imgs/inst_det/teste'+str(i)+'.png', img_parcial)
                    #os.system('./extract_features_64bit.ln -hesaff -hesThres\
                    #        500 -sift -DP -i teste.png')
                    cv2.waitKey(50)

                segm_total = np.multiply(segm_total, img)

                #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                #ret,thresh = cv2.threshold(imgray,127,255,0)
                #im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                
                #cnt = contours[5]
                #x,y,w,h = cv2.boundingRect(cnt)
                #print(x, y, w, h)
                #img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                #img = cv2.rectangle(img_,(x,y),(x+w,y+h),(0,255,0),2)
                #mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                #mask[x:x+w, y:y+h] = 1
                #print(np.nonzero(mask))
                #a = cv2.boxPoints(a)
                #a = np.int0(a)
                #cv2.drawContours(im2, cnt, 0, (0,255,0), 20)
                #img_ = np.multiply(img, mask)
                #print(np.nonzero(img_))

                #cv2.imshow('segm', segm_total)
                #cv2.waitKey(50)

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

