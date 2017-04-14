# -*- coding=UTF-8 -*- 
"""

File name : code_7.py

Creation date : 17-11-2016

Last modified :

Created by :

Purpose :

Usage :

Observations :

    Differences from code_6:

        DONE: Image description is now done with google deep learning network.

        DONE: Saliency peaks are now used instead of absolute salient value of pixels.

        CHANGE: An idea was to use RBD saliency algorithm, since it performs better than Itti's simple 
        algorithm. However, it runs almost 8x slower with no visible gain in our case because we're more 
        interest in salient pixels that will function as seeds. RBD provides salient superpixels.

"""

import os
os.environ['GLOG_minloglevel'] = '3' 
import caffe
import vrep
import cv2
import numpy as np
import time
import sys
import mylib as ml
import bottleneck as bn
import SGNG
from saliency import saliency
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.ndimage as ndimage

def single_feature_extractor(img):

    # Extracts googlenet features from img (real image, np array)
    # Saves the description to imgs/temp

    cv2.imwrite('imgs/temp/temp.jpg', img) 

    # grab args
    img_name = 'imgs/temp/temp.jpg' # absolute path
    #save_dir = sys.argv[2] # absolute path

    cwd = os.getcwd()

    #if not os.path.exists(save_dir):
    #    os.mkdir(save_dir)

    #load the model
    proto = '/home/felipe/caffe/caffe-master/models/bvlc_googlenet/deploy.prototxt'
    model = '/home/felipe/caffe/caffe-master/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    net = caffe.Net(proto,\
                    model,\
                    caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('/home/felipe/caffe/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    #note we can change the batch size on-the-fly
    #since we classify only one image, we change batch size from 10 to 1
    net.blobs['data'].reshape(1,3,224,224)

    img = caffe.io.load_image(img_name)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    feature = net.blobs['pool5/7x7_s1'].data[0].reshape(1, -1)
    #np.save(save_dir + '/' + img_name + '.cnn', feature)
    np.save(img_name + '.cnn', feature)

    return feature[0]


def segmentation_processing(img, segm_total, max_indices, i):

    # performs all the dirty work of image segmentation. returns 
    # the segmented image according to the salient point provided by 
    # max_indices and i

    img_parcial = None

    # checking whether saliency point is in previously grown region
    #print(max_indices)
    if (segm_total[int(max_indices[1][i]), int(max_indices[0][i])] == 0).all():
        # x coordinate is the second element (not sure why...)
        segm_current = ml.stochastic_reg_grow(img, (max_indices[1][i],\
            max_indices[0][i]),\
            30, 30, 20, 30, 60)

        segm_current[np.where(np.mean(segm_current, axis=2) != 0)] = 1
        segm_current = np.asarray(segm_current, dtype=np.bool)
        segm_current = np.multiply(segm_current, img)

        (T, B, L, R) = rectangle_detect(segm_current)
        # t b l r will be None if there is too few or too many pixels in segm_current
        if T is not None:

            segm_current[T:B, L:R] = 1
            segm_current = np.asarray(segm_current, dtype=np.bool)
            segm_total[np.where(np.mean(segm_total, axis=2) != 0)] = 1
            segm_total = np.asarray(segm_total, dtype=np.bool)

            if (np.bitwise_and(segm_current, segm_total) == 0).all():
                segm_current[T:B, L:R] = 1
                segm_current = np.asarray(segm_current, dtype=np.bool)

                # updating segm_total
                segm_total = np.bitwise_or(segm_current, segm_total)
                segm_total = np.multiply(segm_total, img)

                img_parcial = np.multiply(segm_current, img)

                # building box in segmented image, only if pixel_counts are 
                # in between thresholds (see transfere function)
                img_parcial = transfere(img_parcial, img)

    return img_parcial, segm_total

def is_near(pose1, pose2):

    # checks whether pose1 is near pose2 (inside square of side d degrees)

    # d in degrees
    d = 2
    # d in radians
    d = d * np.pi / 180

    b_l = np.asarray(pose1) - np.array([d/2, d/2])
    t_r = np.asarray(pose1) + np.array([d/2, d/2])

    if (pose2[0] >= b_l[0]) and (pose2[1] >= b_l[1]) and \
            (pose2[0] <= t_r[0]) and (pose2[1] <= t_r[1]):
        return 1
    else:
        return 0


def ior_update(ior, pose, y_field, n_field):

    # Updates ior table. Checks and see whether the pose was already 
    # in the table. If it was, it means that the object in that pose 
    # has been changed. This entry is deleted and a new one, corresponding 
    # to the new object, is added.

    for i in range(len(ior)):

        curr_pose = ior[i][0]
        curr_y = ior[i][1]
        curr_n = ior[i][2]
        if is_near(curr_pose, pose):
            ior[i][0] = pose
            ior[i][1] = y_field
            ior[i][2] = n_field
            return ior

    # If it gets here, it means there was no pose near to the 
    # one passed as argument. Lets add it to ior then.

    ior.append([pose, y_field, n_field])
    return ior



def search_mode(sgng, dictionary, last_joint_pos, clientID, v0):

    global ior_table

    print('\n\nEntering search mode.\n')
    # make sure sgng has at least two nodes
    if len(sgng.node_list) < 2:
        print("SGNG must have at least two nodes. Run some exploration first.")
        return 0

    category = None

    while (category not in dictionary):

        category = input('\n\nWhat object category should I look for? (Type in \
\'exploration\' to return to exploration mode.)\n')

        if category == 'exploration':
            return 0
        elif category not in dictionary:
            print('\nCategory %s not in dictionary.' %(category))

    img_parcial = None
    moves = 0

    # Keep moving while max moves not attained
    while moves < max_moves:

        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)
        if res==vrep.simx_return_ok:

            # converting vrep img to numpy array bgr for proper opencv handling
            img = img2rgbnp(image, resolution)
            cv2.imshow('search', img)
            cv2.waitKey(50)

            # First check in ior_table whether the object category has been seen before. 
            # Must check the y_fields of ior_table

            visited_cats = [i[1] for i in ior_table]
            if category in visited_cats:

                while category in visited_cats:

                    ior_index = visited_cats.index(category)
                    obj_pose = ior_table[ior_index][0]

                    # Now moving to obj_pose and grabbing image
                    vrep.simxSetJointTargetPosition(clientID, joint_z,\
                            obj_pose[0],\
                            vrep.simx_opmode_oneshot)

                    vrep.simxSetJointTargetPosition(clientID, joint_x,\
                            obj_pose[1],\
                            vrep.simx_opmode_oneshot)
                    # updating last_joint_pos
                    last_joint_pos = obj_pose


                    cv2.waitKey(1500)

                    res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)
                    img = img2rgbnp(image, resolution)
                    cv2.imshow('search', img)
                    cv2.waitKey(50)

                    # Now processing the seed, which is at the center of the image (256, 256) for 512 res
                    segm_current = ml.stochastic_reg_grow(img, (x_resolution//2,\
                        y_resolution//2),\
                        30, 30, 20, 30, 60)
                    cv2.imshow('aa', segm_current)
                    cv2.waitKey(500)
                    cv2.destroyWindow('aa')

                    segm_current[np.where(np.mean(segm_current, axis=2) != 0)] = 1
                    segm_current = np.asarray(segm_current, dtype=np.bool)
                    segm_current = np.multiply(segm_current, img)

                    (T, B, L, R) = rectangle_detect(segm_current)

                    if T is not None:

                        segm_current[T:B, L:R] = 1
                        segm_current = np.asarray(segm_current, dtype=np.bool)
                        img_parcial = np.multiply(segm_current, img)
                        img_parcial = transfere(img_parcial, img)

                        # Now checking whether segmented image is of desired category

                        #classeme = extract_single_classeme(img_parcial)
                        features = single_feature_extractor(img_parcial)

                        #winner, sec_winner, __ = sgng._bmu(classeme, 0) 
                        winner, sec_winner, __ = sgng._bmu(features, 0) 
                        bmu_label = number2name(dictionary, winner.label)
                        sbmu_label = number2name(dictionary, sec_winner.label)
                        print(bmu_label, sbmu_label)

                        if category in (bmu_label, sbmu_label):
                            
                            cv2.imshow('parc', img_parcial)
                            cv2.waitKey(2000)
                            yn = input('\n\nI think I found it. Is that correct?(y/n)\n\n')
                            if yn == 'y':
                                print('\n')
                                cv2.destroyWindow('parc')
                                cv2.destroyWindow('search')
                                #ior_table = ior_update(ior, obj_pose, category, None)
                                return last_joint_pos
                            elif yn == 'n':
                                ior_table[ior_index][1] = None
                                ior_table[ior_index][2] = category

                            cv2.destroyWindow('parc')

                    # After checking the first pose, update visited_cats and 
                    # verify again if there is category in ior_table
                    visited_cats[ior_index] = -1
            
            # if there is no category in ior_table
            else:

                segm_total = np.zeros((img.shape[0], img.shape[1], 3))

                #sal = get_saliency(img, thresh)
                sal = get_saliency(img)
                max_indices = get_n_max_sal(sal, qty_peaks)

                # Browsing through salient points

                for i in range(qty_peaks):

                    # checking whether saliency point is in previously grown region
                    if (segm_total[max_indices[1][i], max_indices[0][i]] == 0).all():
                        # x coordinate is the second element (not sure why...)
                        segm_current = ml.stochastic_reg_grow(img, (max_indices[1][i],\
                            max_indices[0][i]),\
                            30, 30, 20, 30, 60)

                        segm_current[np.where(np.mean(segm_current, axis=2) != 0)] = 1
                        segm_current = np.asarray(segm_current, dtype=np.bool)
                        segm_current = np.multiply(segm_current, img)

                        (T, B, L, R) = rectangle_detect(segm_current)
                        # t b l r will be None if there is too few or too many pixels in segm_current
                        if T is not None:

                            segm_current[T:B, L:R] = 1
                            segm_current = np.asarray(segm_current, dtype=np.bool)
                            segm_total[np.where(np.mean(segm_total, axis=2) != 0)] = 1
                            segm_total = np.asarray(segm_total, dtype=np.bool)

                            # Checking whether there is intersection between the 
                            # current grown region and the total grown region of the current
                            # scene
                            if (np.bitwise_and(segm_current, segm_total) == 0).all():
                                #segm_current[T:B, L:R] = 1
                                #segm_current = np.asarray(segm_current, dtype=np.bool)

                                # updating segm_total
                                segm_total = np.bitwise_or(segm_current, segm_total)
                                segm_total = np.multiply(segm_total, img)

                                cv2.waitKey(50)
                                img_parcial = np.multiply(segm_current, img)

                                cv2.waitKey(50)
                                # building box in segmented image, only if pixel_counts are 
                                # in between thresholds (see transfere function)
                                img_parcial = transfere(img_parcial, img)
                                cv2.waitKey(50)

                                if img_parcial is not None:

                                    #cv2.waitKey(100)
                                    #cv2.imshow('parc', img_parcial)
                                    #cv2.waitKey(100)

                                    #classeme = extract_single_classeme(img_parcial)
                                    features = single_feature_extractor(img_parcial)

                                    #winner, sec_winner, __ = sgng._bmu(classeme, 0) 
                                    winner, sec_winner, __ = sgng._bmu(features, 0) 
                                    bmu_label = number2name(dictionary, winner.label)
                                    sbmu_label = number2name(dictionary, sec_winner.label)
                                    print(bmu_label, sbmu_label)
                                    #bmu_label = 'clock'
                                    #sbmu_label = 'clock'

                                    if category in (bmu_label, sbmu_label):
                                        
                                        # grabbing mid point, which will be used to calculate the 
                                        # pose
                                        #mid_point = ((B-T)//2, (R-L)//2)
                                        x_joint_pos = get_x_joint_pos(persp_angle, x_resolution, max_indices[1][i],\
                                                last_joint_pos[0])
                                        y_joint_pos = get_y_joint_pos(persp_angle, y_resolution, max_indices[0][i],\
                                                last_joint_pos[1])
                                        obj_pose = (x_joint_pos, y_joint_pos)

                                        cv2.imshow('parc', img_parcial)
                                        cv2.waitKey(2000)
                                        yn = input('\n\nI think I found it. Is that correct?(y/n)\n\n')
                                        if yn == 'y':
                                            print('\n')
                                            cv2.destroyWindow('parc')
                                            cv2.destroyWindow('search')
                                            ior_table = ior_update(ior_table, obj_pose, category, None)
                                            return last_joint_pos
                                        elif yn == 'n':
                                            ior_table = ior_update(ior_table, obj_pose, None, category)

                                        cv2.destroyWindow('parc')
                                        """
                                        last_joint_pos = (0, 0)
                                        x_joint_pos = get_x_joint_pos(np.pi/4, 512, max_indices[1][i],\
                                                last_joint_pos[0])

                                        y_joint_pos = get_y_joint_pos(np.pi/4, 512, max_indices[0][i],\
                                                last_joint_pos[1])

                                        vrep.simxSetJointTargetPosition(clientID, joint_z,\
                                                x_joint_pos,\
                                                vrep.simx_opmode_oneshot)

                                        vrep.simxSetJointTargetPosition(clientID, joint_x,\
                                                y_joint_pos,\
                                                vrep.simx_opmode_oneshot)
                                        """
                # Didnt find in the current scene. Asking permission to move and continue search
                yn = input('\nCouldn\'t find %s in the current scene. Continue search? (y/n)\n' %(category))
                if yn == 'n':
                    cv2.destroyWindow('search')
                    return last_joint_pos
                else:

                    # If continue, move towards the right edge of the screen
                    x_joint_pos = get_x_joint_pos(persp_angle, x_resolution, x_resolution,\
                            last_joint_pos[0])

                    vrep.simxSetJointTargetPosition(clientID, joint_z,\
                            x_joint_pos,\
                            vrep.simx_opmode_oneshot)

                    last_joint_pos = (x_joint_pos, last_joint_pos[1])

                    cv2.waitKey(1000)

                # incrementing moves count
                moves += 1

    # if it gets here, it means that moves has attained max_moves
    print('\nMaximum number of moves attained. Aborting search...\n\n')
    return last_joint_pos


             


def exploration_mode(sgng, dictionary, answer, img):

    # extracting classeme
    print('\n\n Describing image...\n')
    #classeme = extract_single_classeme(img)
    features= single_feature_extractor(img)

    # new category: must find an appropriate number label for sgng
    if answer not in dictionary:

        print('\nNew category. Updating SGNG...\n')
        
        # finding maximum number label
        if len(dictionary.values()) > 0:
            max_value= np.max(np.array([dictionary[i] for i in dictionary]))
        else:
            max_value = -1
        new_label = max_value + 1

        # updating sgng
        #sgng.train_single(classeme, new_label)
        sgng.train_single(features, new_label)

        # updating dictionary
        dictionary[answer] = new_label

        print('\n Done. Looking for new objects...\n')

    # category already exists
    else:

        print('\n\n Category already exists. Updating SGNG...\n')

        # updating sgng
        #sgng.train_single(classeme, dictionary[answer])
        sgng.train_single(features, dictionary[answer])

def number2name(dictionary, number):
    # Receives dictionary whose keys are name, and a number. Returns 
    # the (first) name associated with the number in dict. 
    # If doesnt find, returns none

    for name in dictionary:
        if dictionary[name] == number:
            return name

    return None

def pixel_count(img):
    """
    counts the number of nonzero pixels in img
    """

    h_thres = np.array([255, 255, 255], np.uint8)
    l_thres = np.array([1, 1, 1], np.uint8)

    thres_img = cv2.inRange(img, l_thres, h_thres)
    qty_pixels = cv2.countNonZero(thres_img)

    return qty_pixels

def rectangle_detect(sali):
    """
    recebe img salientada, retorna (L,R,T,B)

    faz analise de quantidade de pixels: se pixel count for 
    menor do que 4% da area da imagem ou se for maior do 
    40% da area da imagem, ignora segmentaçao
    """
    
    mult = 1
    width = sali.shape[1]
    height = sali.shape[0]
    offset_x = 1
    offset_y = 1

    #pixel_count = 0
    l_thres = 0.005*sali.shape[0]*sali.shape[1]
    h_thres = 0.05*sali.shape[0]*sali.shape[1]
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
    #print(pixel_count, l_thres, h_thres)
    #print(cv2.countNonZero(sali))
    p_count = pixel_count(sali)
    #print(p_count, l_thres, h_thres)
    if (p_count > l_thres) and (p_count < h_thres):
        return (T, B, L, R)
    else:
        return (None, None, None, None)

def transfere(sali, orig):

    """
    recebe imagem salientada, original e retorna imagem redimensionada de
    acordo com a bounding box
    """

    box = rectangle_detect(sali)
    #print(box[0], box[1], box[2], box[3])
    if box[0] is not None:
        cropped = orig[box[0]:box[1],box[2]:box[3]]
        return cropped
    else:
        return None

def get_saliency(img):

    #Calculates img saliency, using NVT algorithm (Itti, 1998)

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

#def get_saliency(img, thresh):
"""
def get_saliency(img):

    # Calculates image saliency using RBD visual attention algorithm. 
    # OBS.: the executable rbd/run_my_rbd.sh must be running with the 
    # following command:  ./run_my_rbd.sh $MCR_dir $temp_dir/image_file_name.txt
    #
    # $MCR_dir is the directory that contains matlab compiler runtime (it's free to download),
    # $temp_dir is the global temp_dir variable in this script, where temporary images will be 
    # saved to,
    # image_file_name.txt is a text file containing the name of the image to be 
    # processed. It must be named 'image_file_name.txt'.
    # 
    # The attention algorithm runs as soon as there is a change in image_file_name.txt
    # The algorithm outputs a binarized image. 

    # Grabbing a random image name
    img_name = str(np.int(10000*np.random.random())) + '.jpg'
    cv2.imwrite(temp_dir + '/' + img_name, img)

    f = open(temp_dir + '/image_file_name.txt', 'w')
    f.write(img_name + '\n')
    f.close()

    # waiting for the saliency algorithm to run
    cv2.waitKey(500)

    # grabbing the saliency image from generated file
    sal_name = temp_dir + '/' + img_name[:-4] + '_saliency.png'
    sal = cv2.imread(sal_name)

    sal = sal[:, :, 0]
    #sal = sal.astype(np.uint8)

    os.remove(sal_name)
    os.remove(temp_dir + '/' + img_name)

    return sal 
""" 

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


def get_n_max_sal(sal, qty_peaks):

    #
    #Receives:
    #    saliency map
    #    n, number of points to return

    #Returns:
    #    numpy array with the indices of maximum points
    #

    filter_size = 20

    pooled_data = ndimage.maximum_filter(sal, size=filter_size)
    maxima = (sal == pooled_data)
    indices = np.where(maxima)
    #max_indices = np.argsort(data[indices])[-qty_peaks:]
    max_indices = np.argsort(sal[indices])[-np.min([qty_peaks, len(indices[0])]):]
    #print(len(max_indices))
    #print(len(max_indices) / qty_peaks)

    final_indices = []
    #for i in range(qty_peaks):
    for i in range(len(max_indices)):
        final_indices.append([indices[0][max_indices[i]], indices[1][max_indices[i]]])
        #assert indices[0][max_indices[i]] < data.shape[0], 'eita'
        #assert indices[1][max_indices[i]] < data.shape[1], 'eita'

    final_indices = np.array(final_indices)
    final_indices = final_indices.T

    return final_indices


"""
def get_n_max_sal(sal, n):

    #
    #Receives:
    #    saliency map
    #    n, number of points to return

    #Returns:
    #    numpy array with the indices of maximum points
    #

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
        max_indices[0][i] = int(max_curr_ind[0])
        max_indices[1][i] = int(max_curr_ind[1])

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
"""

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
    res = vrep.simxSetObjectPosition(clientID, model_handle, -1, (-0.8, 0, 5), vrep.simx_opmode_oneshot)

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

                
                """
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

######

# GLOBAL VARIABLES
 
######

# number of most salient points to start segmentation
global qty_peaks
qty_peaks = 4

# camera resolution
global x_resolution
x_resolution = 512

global y_resolution
y_resolution = 512

# camera perspective angle in rad
global persp_angle
persp_angle = np.pi / 4

# maximum number of moves until object is declared not found
global max_moves
max_moves = 4

# inhibition of return table
global ior_table
ior_table = [[-1, -1, -1]]

# directory where temporary images will be saved
global temp_dir 
temp_dir = 'imgs/temp/'

# binarization threshold for saliency
#global thres
#thresh = 0.5

# checking whether to begin a new agent or load a saved one
# obs.: the sgng file must be in npy format
# if it is to be loaded, the dictionary file must be loaded as well
if sys.argv[1] == 'new':
    sgng = SGNG.SGNG()
    dictionary = {}
else:
    sgng = np.load(sys.argv[1])[0]
    dictionary = np.load(sys.argv[2])[0]

# close all open connections

vrep.simxFinish(-1)

# start new connection.
# IMPORTANT: there must be the following line in a THREADED child script in vrep:
# simExtRemoteApiStart(19999)

#print(is_near((np.pi/4, np.pi/4), (np.pi/4, np.pi/4+0.01)))

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

#analyze_model(clientID, '/home/felipe/vrep/clock2/clock2.ttm', 15)

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
                #sal = get_saliency(img, thresh)
                begin = time.time()
                sal = get_saliency(img)

                # finding the most salient point
                #max_sal_point = np.unravel_index(np.argmax(sal), sal.shape)

                # finding the N most salient points

                max_indices = get_n_max_sal(sal, qty_peaks)
                end = time.time() - begin
                print(end)
                max_sal_point = (max_indices[0][0], max_indices[1][0])

                #cv2.imshow('static sal', sal)
                #cv2.waitKey(500)

                # initializing segmentation
                segm_total = np.zeros((resolution[0], resolution[1], 3))

                # iterating over all salient points
                for i in range(qty_peaks):

                    img_parcial, segm_total = segmentation_processing(img, segm_total, max_indices, i)

                    if img_parcial is not None:

                        #cv2.imshow('segm', segm_total)
                        cv2.waitKey(50)
                        cv2.imshow('segm', img_parcial)
                        cv2.waitKey(50)

                        # Beginning interaction
                        answer = input('\nWhat is it? (Type in \'search\' to toggle search mode)\n')
                        if answer != 'search':
                            exploration_mode(sgng, dictionary, answer, img_parcial) 
                        if answer == 'search':
                            last_joint_pos = search_mode(sgng, dictionary, last_joint_pos, clientID, v0)
                            break

                        #cv2.imwrite('imgs/test/teste'+str(i)+'.png', img_parcial)

                #segm_total = np.multiply(segm_total, img)

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

                x_joint_pos = get_x_joint_pos(persp_angle, x_resolution, max_sal_point[1],\
                        last_joint_pos[0])

                y_joint_pos = get_y_joint_pos(persp_angle, y_resolution, max_sal_point[0],\
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

