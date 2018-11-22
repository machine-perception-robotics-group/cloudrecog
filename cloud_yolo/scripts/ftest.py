#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import chainer
from chainer import serializers
from chainer import Variable
from facialattr_vgg.structure import network
import numpy as np
import argparse
import cPickle as pickle

import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='face.png')
    args = parser.parse_args()
    # f = open("img.pkl","r")
    # d = pickle.load(f)
    color_img = cv2.imread(args.image)
    img_shape = color_img.shape
    img = color_img.astype(np.float32)
    img = cv2.resize(img, (227, 227))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255
    print "image: %s" %(args.image)

    net = network()
    serializers.load_hdf5('facialattr_vgg_tune/epoch-100.model',net)
    print img.shape
    img = img[np.newaxis, np.newaxis, :, :]
    pred = net(Variable(img),0,14)
    #pred = net(Variable(pred.data), 1, 2)
    pos = pred[0].data[0]
    print pos
    print pred[1].data*66
    print pred[2].data*100
    gender = np.argmax(pred[3].data)

    if gender == 1:
        gender_class = 'Male'
    else:
        gender_class = 'Female'
    print gender_class

    race = np.argmax(pred[4].data)
    if race == 0:
        race_class = 'Asian'
    if race == 1:
        race_class = 'White'
    if race == 2:
        race_class = 'Black'
    print race_class
    point = np.zeros((5, 2))
    for c in range(5):
            for i in range(2):
                point[c][i]=pos[c*2+i]*img_shape[0]
    print point
    point_scale = 3
    for x, y in point:
        x = int(round(0 + 1 * x))
        y = int(round(0 + 1 * y))
        cv2.circle(color_img, (x, y), point_scale, (0, 255, 0), -1)
        cv2.circle(color_img, (x, y), point_scale, (0, 0, 0), 1)

    cv2.imwrite("hoge.png",color_img)






if __name__ == '__main__':
    main()
