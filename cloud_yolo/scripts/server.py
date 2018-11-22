#!/usr/bin/env python
# -*- coding: utf-8 -*-

import roslib
# roslib.load_manifest('ros_chainer_base')
import sys
import rospy
import numpy as np
import argparse


from cloud_yolo.srv import *
#from model.structure import network
from yolov2 import *
from chainer import serializers, Variable, cuda
import cPickle as pickle
# import time

import modules.npm_conv as npm_conv
import time

class ServiceProvider:
    def __init__(self, args):
        self.gpu = args.gpu
        self.n_classes = 20
        self.n_boxes = 5
        yolov2 = YOLOv2(n_classes=self.n_classes, n_boxes=self.n_boxes)
        model = YOLOv2Predictor(yolov2)

        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            model.to_gpu(self.gpu)
        print "Loading Model..."
        serializers.load_hdf5(args.model, model)
        self.model = model
        self.service = rospy.Service('DCNN_CalcService', DCNN_uint8, self.callback, buff_size=65536)

    def callback(self, req):
        #print req
        rospy.loginfo("div_layer: {}, len:{}".format(req.start_layer, len(req.input)))

        data = pickle.loads(req.input)
        
        start = time.time()

        if self.gpu >= 0:
            if data["hf"] is not None:
                hf = cuda.cupy.asarray(data["hf"].data, dtype=np.float32)
                self.model.predictor.high_resolution_feature = Variable(hf)
            data = cuda.cupy.asarray(data["fmap"], dtype=np.float32)
        
        result = self.model.predict_outputside(Variable(data), req.start_layer)
        rospy.loginfo("Time: %f [ms]", (time.time() - start) * 1000)
        if self.gpu >=0:
            result.data = cuda.to_cpu(result.data)
            output = {"data": result.data, "ptime": time.time()-start}
            response = pickle.dumps(output, pickle.HIGHEST_PROTOCOL)

            return DCNN_uint8Response(response)


def main():
    rospy.init_node('dcnn_ServiceServer')
    parser = argparse.ArgumentParser(description='ROS-Chainer demo Server Script')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', type=str, default="facialattr_vgg/epoch-100.model")
    args = parser.parse_args(rospy.myargv()[1:])

    s = ServiceProvider(args)
    print "waiting call"

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
