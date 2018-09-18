#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
from mobilenet_chainer import predictor
import os
import chainer
import numpy as np
import chainer
import cPickle as pickle
import time
import rospy
import roslib

import argparse
import time
import sys
import signal
import cPickle as pickle
from cloud_mobilenet.srv import *
import numpy as np


class ServiceServer():
    def __init__(self, args):
        self.gpu = args.gpu
        self.model = predictor("model_iter_232600", args.gpu)

        srv = rospy.Service('recog_srv', recog_msgs, self.callback)
        print("GPU STATUS {}".format(self.gpu))
        print("waiting call")

    def callback(self, req):
        start = time.time()
        input_data = pickle.loads(req.input)

        y = self.model.resume(input_data, req.start_layer)
        output = {"data": y, "ptime": time.time()-start}
        output = pickle.dumps(output, pickle.HIGHEST_PROTOCOL)

        return recog_msgsResponse(output)

def main():
    rospy.init_node('recog_srv')
    parser = argparse.ArgumentParser(description='ROS-Chainer demo Server Script')
    parser.add_argument('--gpu', type=int, default=-1)
    # parser.add_argument('--model', type=str, default='facialattr_vgg/epoch-100.model')
    args = parser.parse_args(rospy.myargv()[1:])
    s = ServiceServer(args)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()