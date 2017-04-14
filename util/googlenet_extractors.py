# -*- coding=UTF-8 -*-

"""

File name : googlenet_extractors.py

Creation date : 08-04-2017

Last modified :

Created by :

Purpose :

    Routines for googlenet features extraction. Includes: network initialization, bulk and single extractors.

Usage :

    First call initialize_googlenet (make sure to edit proto and model to the correct path), then call 
    either bulk_ or single_feature_extractor. 

Observations :

"""

def bulk_feature_extractor(imgs_list):

    # Extracts feature from all imgs from list
    # List contains images file names

    # Returns matrix with features

    batch_size = len(imgs_list)
    net.blobs['data'].reshape(batch_size,3,224,224)

    preprocessed_data = np.array([transformer.preprocess('data', caffe.io.load_image(x)) for x in imgs_list])

    net.blobs['data'].data[...] = preprocessed_data
    out = net.forward()


    features_matrix = net.blobs['pool5/7x7_s1'].data
    features_matrix = np.reshape(features_matrix, (batch_size, features_matrix.shape[1]))

    return features_matrix 





def initialize_googlenet():

    # Performs some network configuration on googlenet

    global net
    global transformer


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
    #net.blobs['data'].reshape(1,3,224,224)

    return net, transformer

#def single_feature_extractor(img, net, transformer):
def single_feature_extractor(img):

    # Extracts googlenet features from img (real image, np array)
    # Saves the description to imgs/temp

    # Returns features 1-D array

    #begin_overhead = time.time()

    #begin_feat = time.time()

    temp_dir = '~/temp'

    img_name = str(np.int(100000*np.random.random())) + '.jpg'
    img_name = temp_dir + '/' + img_name # absolute path
    #cv2.imwrite(temp_dir + '/' + img_name, img)
    cv2.imwrite(img_name, img)

    #cv2.imwrite(temp_dir + '/temp.jpg', img) 

    # grab args
    #img_name = temp_dir + '/temp.jpg' # absolute path
    #save_dir = sys.argv[2] # absolute path

    #cwd = os.getcwd()

    img = caffe.io.load_image(img_name)

    # making batch size equal to 1, single it's single extractor
    net.blobs['data'].reshape(1,3,224,224)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    out = net.forward()
    feature = net.blobs['pool5/7x7_s1'].data[0].reshape(1, -1)

    np.save(img_name + '.cnn', feature)
    #end_feat = time.time() - begin_feat
    #os.remove(temp_dir + '/' + img_name)
    #os.remove(img_name)
    #print(end_feat)

    return feature[0]


