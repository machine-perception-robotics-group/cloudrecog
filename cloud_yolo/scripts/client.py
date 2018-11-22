#!/usr/bin/env python
# -*- coding: utf-8 -*-

import roslib
# roslib.load_manifest('ros_chainer_base')
import sys
import rospy
import cv2
import math
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import argparse
import threading
from cloud_yolo.srv import *
from cloud_yolo.msg import *
# from model.structure import network
from chainer import serializers, Variable
from sensor_msgs.msg import Image
import time
from modules.piece import GammaCorrect
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)
import time
import cPickle as pickle
from yolov2 import *
from lib.utils import *
import random


class Img:
    def __init__(self, img):
        self.__img = img

    def getimg(self):
        return self.__img

    def setimg(self, img):
        self.__img = img

class Getimage:
    def __init__(self, im_class):
        self.__im_class = im_class
        self.image = None
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image_raw", Image, self.img_callback)

    def img_callback(self, img_dat):
        """
        画像トピックの購読用callback
        """
        try:
            self.image = self.cv_bridge.imgmsg_to_cv2(img_dat, "bgr8")
        except CvBridgeError as e:
            # ROS_ERROR(e)
            pass
        self.__im_class.setimg(self.image)


class YoloPredictor:
    def __init__(self, im_class, args):
        # div Layerをrosparamから設定するように変更 11/13
        # self.divlayer = args.div
        if rospy.has_param('div_layer'):
            self.div_layer = int(rospy.get_param('div_layer'))
        else:
            self.div_layer = 2

        self.__im_class = im_class

        self.srv_msg = rospy.ServiceProxy('/DCNN_CalcService', DCNN_uint8)  # DCNN実行サービス
        self.pub = rospy.Publisher('image_out', Image, queue_size=1) # 出力結果のPublisher
        self.pub_time = rospy.Publisher('processing_time', processing_time, queue_size=1) #各処理時間のPublisher
        self.cv_bridge = CvBridge()
        self.n_classes = 20
        self.n_boxes = 5
        self.gpu = args.gpu
        self.detection_thresh = 0.3
        self.iou_thresh = 0.3
        self.gamma_correct = GammaCorrect(1.8)
        self.labels = load_label_file("yolo/voc.names")
        yolov2 = YOLOv2(n_classes=self.n_classes, n_boxes=self.n_boxes)
        model = YOLOv2Predictor(yolov2)
        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            model.to_gpu(self.gpu)
        serializers.load_hdf5(args.model, model)  # load saved model
        model.predictor.train = False
        model.predictor.finetune = False
        self.model = model
        print("client started.")

        # rospy.wait_for_service('DCNN_CalcService')
        while True:
            # th = threading.Thread(target=self.process, name="fs", args=())
            self.im = self.__im_class.getimg()
            if self.im is not None:
                start = time.time()
                nms_results = self.prediction(self.im)
                pt = time.time() - start

                self.draw(self.im, nms_results, pt)

    def prediction(self, orig_img):
        # パラメータ問い合わせ
        if rospy.has_param('div_layer'):
            self.div_layer = int(rospy.get_param('div_layer'))

        orig_input_height, orig_input_width, _ = orig_img.shape

        img = reshape_to_yolo_size(orig_img)
        input_height, input_width, _ = img.shape
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # forward
        x_data = img[np.newaxis, :, :, :]
        start = time.time()
        # GPUの場合はinputをGPUへ転送
        if self.gpu >= 0:
            x_data = cuda.cupy.asarray(x_data, dtype=np.float32)
        x = Variable(x_data)

        # predict
        # print dir(self.model.predictor)
        pred = self.model.predict_inputside(x, self.div_layer)
        timeof_c = time.time() - start

        hf = self.model.predictor.high_resolution_feature

        # 配列をpickleで符号化
        data = pickle.dumps({"fmap": pred.data, "hf": hf}, protocol=pickle.HIGHEST_PROTOCOL)

        # Serverにそうしん〜〜
        start = time.time()
        while True:
            try:
                recv = self.srv_msg(self.div_layer, data)
                break
            except rospy.ServiceException as exc:
                pass
        timeof_ct = time.time() - start
        # print res

        # 漬物をもどす
        recv = pickle.loads(recv.output)
        self.pub_time.publish(str(self.div_layer), timeof_c, recv["ptime"], (timeof_ct - recv["ptime"]))
        # rospy.loginfo(
        #     "{:.3f},{:.3f},{:.3f}".format(timeof_c * 1000, recv["ptime"] * 1000, (timeof_ct - recv["ptime"]) * 1000))
        res = recv["data"]
        x, y, w, h, conf, prob = self.model.calc_bbox(res)

        _, _, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (self.n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (self.n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (self.n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (self.n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (self.n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (self.n_boxes, self.n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh

        results = []
        for i in range(detected_indices.sum()):
            results.append({
                "label": self.labels[prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()],
                "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
                "conf": conf[detected_indices][i],
                "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
                "box": Box(
                    x[detected_indices][i] * orig_input_width,
                    y[detected_indices][i] * orig_input_height,
                    w[detected_indices][i] * orig_input_width,
                    h[detected_indices][i] * orig_input_height).crop_region(orig_input_height, orig_input_width)
            })

        # nms
        nms_results = nms(results, self.iou_thresh)

        return nms_results

    def draw(self, img, nms_results, pt):
        cv2.putText(img, str(1 / pt), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for result in nms_results:
            left, top = result["box"].int_left_top()
            cv2.rectangle(
                img,
                result["box"].int_left_top(), result["box"].int_right_bottom(),
                (255, 0, 255),
                2
            )
            text = '%s(%2d%%)' % (result["label"], result["probs"].max() * result["conf"] * 100)
            cv2.putText(img, text, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # print(text)
        try:
            self.pub.publish(self.cv_bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)

        # cv2.imwrite("hoge.png", img)
        # cv2.imshow("hoge", img)
        # cv2.waitKey(0)


def main():
    rospy.init_node('ros_chainer_base_{}'.format(random.randint(1, 1000)), anonymous=True)
    parser = argparse.ArgumentParser(description='ROS-Chainer demo Client Script')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', type=str, default='facialattr_vgg/epoch-100.model')
    parser.add_argument('--div', type=int, default=0)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.loginfo("GPU STATUS:%s", args.gpu)
    rospy.loginfo("MODEL:%s", args.model)

    im_class = Img(None)

    th_callback = threading.Thread(target=Getimage, name="th_callback", args=(im_class,))
    th_callback.start()

    # print "im"

    th_ft = threading.Thread(target=YoloPredictor, name="hoge", args=(im_class, args,))
    th_ft.start()

    while th_ft.isAlive():
        # print "alive"
        time.sleep(0.5)
    th_ft.join()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
