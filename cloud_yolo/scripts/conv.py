#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import h5py
import cPickle as pickle
from chainer import serializers,Variable
from face import net
import numpy
import cv2
path = './cnn9_pkl/'


def main():
    netw = net()

    weight={
        "conv1":"00_Conv_16x1x3x3_weight.pkl",
        "conv2":"01_Conv_16x8x3x3_weight.pkl",
        "conv3":"02_Conv_32x8x3x3_weight.pkl",
        "conv4":"03_Conv_32x16x3x3_weight.pkl",
        "conv5":"04_Conv_64x16x3x3_weight.pkl",
        "conv6":"05_Conv_64x32x3x3_weight.pkl",
        "conv7":"06_Conv_128x32x3x3_weight.pkl",
        "conv8":"07_Conv_128x64x2x2_weight.pkl",
        "conv9":"08_Conv_256x64x2x2_weight.pkl",
        "fc1":"09_Full_512x200_weight.pkl",
        "fc2":"10_Full_200x17_weight.pkl"
    }

    bias={
        "conv1":"00_Conv_16_bias.pkl",
        "conv2":"01_Conv_16_bias.pkl",
        "conv3":"02_Conv_32_bias.pkl",
        "conv4":"03_Conv_32_bias.pkl",
        "conv5":"04_Conv_64_bias.pkl",
        "conv6":"05_Conv_64_bias.pkl",
        "conv7":"06_Conv_128_bias.pkl",
        "conv8":"07_Conv_128_bias.pkl",
        "conv9":"08_Conv_256_bias.pkl",
        "fc1":"09_Full_200_bias.pkl",
        "fc2":"10_Full_17_bias.pkl"
    }
    for name in weight.keys():
        print name
        d = pkl_load(path+weight[name])
        print d.shape
        if name == "fc1":
            d=d.T
            logging_shape(name,netw[name].W.data,d)
            netw[name].W.data = d
        elif name == "fc2":
            d = d.T
            logging_shape("fpos", netw["fpos"].W.data, d[0:10])
            netw["fpos"].W.data = d[0:10]

            logging_shape("fpos",netw["gender"].W.data, d[10:12])
            netw["gender"].W.data = d[10:12]

            logging_shape("fpos",netw["age"].W.data, d[12:13])
            netw["age"].W.data = d[12:13]

            logging_shape("fpos",netw["race"].W.data, d[13:16])
            netw["race"].W.data = d[13:16]

            logging_shape("fpos",netw["smile"].W.data, d[16:17])
            netw["smile"].W.data = d[16:17]
        else:
            d = d[:,:,::-1,::-1]
            logging_shape(name,netw[name].W.data,d)
            netw[name].W.data = d


    for name in bias.keys():
        print name
        d = pkl_load(path+bias[name])
        if name == "fc1":
            logging_shape(name, netw[name].b.data,d)
            netw[name].b.data = d
        elif name == "fc2":
            logging_shape("fpos",netw["fpos"].b.data, d[0:10])
            netw["fpos"].b.data = d[0:10]
            logging_shape("fpos",netw["gender"].b.data, d[10:12])
            netw["gender"].b.data = d[10:12]
            logging_shape("fpos",netw["age"].b.data, d[12:13])
            netw["age"].b.data = d[12:13]
            logging_shape("fpos",netw["race"].b.data, d[13:16])
            netw["race"].b.data = d[13:16]
            logging_shape("fpos",netw["smile"].b.data, d[16:17])
            netw["smile"].b.data = d[16:17]
        else:
            logging_shape(name, netw[name].b.data,d)
            netw[name].b.data = d

    print dir(netw.serialize)
    img = cv2.imread("face_.png").astype(numpy.float32)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = img.transpose()
    #img = cv2.resize(img, (100, 100)).transpose((2, 0, 1))
    img = img[numpy.newaxis,numpy.newaxis,:,:]
    print img.shape
    pred = netw(Variable(img),None)

    gender = numpy.argmax(pred[1].data)
    if gender == 1:
        print "Male"
    else:
        print "Female"

    print "age:%d" %(int(round(pred[2].data[0]*66)))
    print "smile:%d" %(int(round(pred[4].data[0]*100)))
    print pred[4].data[0]
    serializers.save_hdf5('facial.model',netw )


def logging_shape(name,a,b):
    print "netw[%s].W.shape:%15s <-- d.shape:%15s" % (name,a.shape,b.shape)


def visualize(filter):
    pass

def pkl_load(fname):
    with open(fname,"r") as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    main()
