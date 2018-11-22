#!/usr/bin/env python
# -*- coding: utf-8 -*-

import roslib
#roslib.load_manifest('ros_chainer_base')
import sys
import rospy
import cv2
import math
import numpy as np
from cv_bridge import CvBridge,CvBridgeError
import argparse
import threading
from ros_chainer_base.srv import *
#from chainer_conf import network
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from sensor_msgs.msg import Image
import time
import signal
import chainer
from chainer import serializers
from chainer import Variable
from facialattr.structure2 import network

signal.signal(signal.SIGINT, signal.SIG_DFL)

#from VGGNet import VGGNet
debug = False


def dump(text):
    """
    デバグ用print function
    """
    if debug:
        print text

class img:
    def __init__(self,img):
        self.__img = img
    def getimg(self):
        return self.__img
    def setimg(self,img):
        self.__img = img


class getimage:
    def __init__(self,im_class):
        self.__im_class = im_class
        self.image = None
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/facialcam/image_raw",Image,self.img_callback)
        #rospy.Subscriber("transmission_time", Float32MultiArray,self.transback )#通信時間測定サービス
        self.srv_msg = rospy.ServiceProxy('DCNN_CalcService',DCNN_MultiArray)#DCNN下位層実行サービス

        dump("init getimage")
    def img_callback(self,img_dat):
        """
        画像トピックの購読用callback
        """
        #dump("img callback!")
        try:
            self.image = self.cv_bridge.imgmsg_to_cv2(img_dat,"bgr8")
        except CvBridgeError as e:
            ROS_ERROR(e)
        self.__im_class.setimg(self.image)



class FaceThread:
    def __init__(self,im_class):
        dump("init FT")
        self.__im_class = im_class
        self._cascade_path = './haarcascades/haarcascade_frontalface_alt.xml'
        self.network = network()
        serializers.load_hdf5('facialattr/epoch-400.model',self.network)
        while True:
            #th = threading.Thread(target=self.process, name="fs", args=())
            self.process()

    def process(self):
        dump("process")
        im = self.__im_class.getimg()
        print "process"
        if im is not None:
            _frame_gray = cv2.cvtColor(im, cv2.cv.CV_BGR2GRAY)

            #カスケード分類器の特徴量を取得する
            _cascade = cv2.CascadeClassifier(self._cascade_path)

            #物体認識（顔認識）の実行
            facerect = _cascade.detectMultiScale(_frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))

            if len(facerect) > 0:
                print '顔が検出されました。'
                _color = (255, 255, 255) #白
                for rect in facerect:
                    #検出した顔を囲む矩形の作成
                    rect[0] = rect[0]  - rect[2] * 0.1
                    rect[1] = rect[1] + rect[3] * 0.01
                    rect[2] = rect[2] * 1.15
                    rect[3] = rect[3] * 1.2

                    img = self.get_proc_image(im, rect, (100, 100))
                    img = img[np.newaxis,np.newaxis,:,:]

                    pred = self.network(Variable(img), 0, 10)

                    point = np.zeros((5, 2))
                    for c in range(5):
                            for d in range(2):
                                point[c][d] = pred[0].data[0][c*2+d]
                    print point
                    point_scale = 2

                    left, top, width, height = rect
                    for x, y in point:
                        x = int(round(left + width * x))
                        y = int(round(top + height * y))
                        cv2.circle(im, (x, y), point_scale, (0, 255, 0), -1)
                        cv2.circle(im, (x, y), point_scale, (0, 0, 0), 1)

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

                    print round(pred[2].data[0]*100)

                    cv2.rectangle(im, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), _color, thickness=2)




            cv2.imshow("hoge",im)
            cv2.waitKey(10)



    def get_proc_image(self, image, rect, input_shape):
        ROI_L = rect[0]
        ROI_T = rect[1]
        ROI_R = rect[0]+rect[2]
        ROI_B = rect[1]+rect[3]

        image = image[ROI_T:ROI_B, ROI_L:ROI_R]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        crop_image = cv2.resize(image, (100,100))
        crop_image = cv2.resize(image, input_shape[-2:], interpolation=cv2.INTER_NEAREST)
        cv2.imshow("roi",crop_image)
        crop_image = np.asarray(crop_image).astype('float32')
        crop_image /= 255.

        return crop_image.reshape(input_shape)

def main():
    rospy.init_node('ros_chainer_base', anonymous=True)

    im_class = img(None)
    windowname = "CV2"
    cv2.namedWindow(windowname)
    th_callback = threading.Thread(target=getimage, name="th_callback", args=(im_class, ))
    th_callback.start()

    print threading.activeCount()
    print "im"

    th_ft = threading.Thread(target=FaceThread, name="hoge", args=(im_class, ))
    th_ft.start()

    while th_ft.isAlive():
        print "alive"
        time.sleep(0.5)
    th_ft.join()


    # while True:
    #     #print "loop"
    #     #frame = th_callback.get_image()
    #     im = im_class.getimg()
    #     if im is not None:
    #         cv2.imshow(windowname,im)
    #     #print threading.activeCount()
    #         if(threading.activeCount() < 2 ):
    #             th = FaceThread(im)
    #             th.start()
    #
    # 	#10msecキー入力待ち
    # 	k = cv2.waitKey(10)
    # 	#Escキーを押されたら終了
    # 	if k == 27:
    # 		break


    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
