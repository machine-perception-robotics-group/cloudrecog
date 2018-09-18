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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import threading
import argparse
import time
import sys
import signal
import cPickle as pickle
from cloud_mobilenet.srv import *
# print(cv2.__version__[0])


if int(cv2.__version__[0]) > 2:
    CC = cv2.LINE_AA
else:
    CC = cv2.CV_AA
#roslib.load_manifest('cloud_resnet_client')

GPU_FLAG=-1

def signal_handler(signal, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# for k, v in os.environ.items():
#     print("{key} : {value}".format(key=k, value=v))
class Image_store():
    def __init__(self):
        self.img = None

    def setimg(self, img):
        self.img = img

    def getimg(self):
        return self.img.copy()

    def getstatus(self):
        if self.img is None:
            return False
        else:
            return True


class LatestImage(threading.Thread):
    def __init__(self, imgs):
        threading.Thread.__init__(self)
        print("init ok")
        self.image_store = imgs
        # self.lock = threading.Lock()
        self.cv_bridge = CvBridge()
        self.cv_image = None


    def run(self):
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_subscriber)
        rospy.spin()

    def image_subscriber(self, img):
        # print("call back")

        try:
            self.cv_image = self.cv_bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.image_store.setimg(self.cv_image)
        # self.event.wait()


class RecogProcess(threading.Thread):
    def __init__(self, imclass, args):
        threading.Thread.__init__(self)
        self.imclass = imclass
        self.gpu = args.gpu
        self.div_layer = args.div
        # model=VGG16Layers()
        self.model = predictor("model_iter_232600", self.gpu)
        # print self.model.available_layers
        # print("GPU STATUS {}".format(self.gpu))

        # rospy.loginfo("GPU STATUS {}".format(self.gpu))
        # if self.gpu > -1:
        #     chainer.cuda.get_device_from_id(self.gpu).use()
        #     self.model.to_gpu()
        #     rospy.loginfo("to_gpu")
        # rospy.loginfo("div:{}".format(self.div_layer))

        self.sclient = rospy.ServiceProxy('recog_srv', recog_msgs)

    def run(self):
        while True:
            start_time = time.time()


            if not self.imclass.getstatus():
                print("Wait for Image")
            else:
                img = self.imclass.getimg()
                y = self.model.extract(img, self.div_layer)
                start = time.time()


                data = pickle.dumps(y, protocol=pickle.HIGHEST_PROTOCOL)

                curr_time = time.time()
                client_time = curr_time-start_time

                res = self.sclient(self.div_layer, data)

                # print(dir(y))
                res = pickle.loads(res.output)
                y = res["data"]
                server_time = res["ptime"]
                transmit_time = time.time()-curr_time-server_time
                # y = self.model.resume(pred[div_layer].data, div_layer)

                # print y.shape
                # print((time.time()-start)*1000)
                clsID = pickle.load(open(
                    'imagenet1000_clsid_to_human.pkl', 'r'))
                cls = clsID[int(
                chainer.functions.argmax(chainer.functions.softmax(y, axis=1), axis=1).data)]
                # print clsID[int(chainer.cuda.to_cpu(chainer.functions.argmax(chainer.functions.softmax(pred['prob'], axis=1), axis=1).data[0]))]
                # print int(chainer.cuda.to_cpu(chainer.functions.argmax(chainer.functions.softmax(pred['prob'], axis=1), axis=1).data[0]))

                rospy.loginfo("Process Time C/S/T(ms): {:.2f}/{:.2f}/{:.2f}".format(client_time*1000, server_time*1000, transmit_time*1000))
                fps = "FPS:{:.1f}".format(float(1.0/(time.time()-start)))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, fps, (10, 30), font, 1, (255, 255, 255), 2, CC)
                cv2.putText(img, cls, (10, 465), font, 1, (255, 255, 255), 1, CC)
                cv2.imshow('demo', img)
                cv2.waitKey(1)
            # print((time.time()-start)*1000)
            # time.sleep(1)


def main():
    rospy.init_node('ros_chainer_base', anonymous=True)
    parser = argparse.ArgumentParser(description='ROS-Chainer demo Client Script')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--div', type=str, default='conv_bn_3')
    # parser.add_argument('--model', type=str, default='facialattr_vgg/epoch-100.model')
    args = parser.parse_args(rospy.myargv()[1:])

    I = Image_store()
    # R = RecogProcess()
    # L = LatestImage(I)
    # L.setDaemon(True)
    # L.start()

    threads = []

    th1 = LatestImage(I)
    th1.daemon = True
    th1.start()

    threads.append(th1)

    th2 = RecogProcess(I, args)
    th2.daemon = True
    th2.start()

    threads.append(th2)

    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()