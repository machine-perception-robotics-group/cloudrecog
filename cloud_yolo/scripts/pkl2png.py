#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cPickle as pickle
import cv2

file="img.pkl"
with open(file,"r") as f:
    d = pickle.load(f)

for i,k in enumerate(d):
    print i
    cv2.imshow("i",k)
    cv2.imwrite("png/"+str(i)+".png",k)

    cv2.waitKey(1)