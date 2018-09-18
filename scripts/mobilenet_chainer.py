#!/usr/bin/env python
# -*- Coding: utf-8 -*-

from mobilenet import MobileNet

import numpy as np
import chainer
import chainer.links as L
import time
import cv2

import json
print(chainer.__version__)


class predictor:
    def __init__(self, initmodel, gpu):
        self.model = L.Classifier(MobileNet())
        chainer.serializers.load_npz(initmodel, self.model)
        self.gpu = gpu
        if self.gpu >= 0:
            chainer.cuda.Device(self.gpu).use()
            self.model.to_gpu()

        self.mean = np.load("mean.npy")
        self.mean = cv2.resize(self.mean.transpose((1, 2, 0)), (224, 224))

    def extract(self, img, div_layer):
        img = cv2.resize(img, (224, 224))
        img = np.asarray(img, dtype=np.float32)

        img-= self.mean
        # PILformat
        img = img[:, :, ::-1]
        # cv2.imshow("",img.astype(np.uint8))
        # cv2.waitKey(0)
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, :]/255
        # print(img.shape)
        if self.gpu >= 0:
            img = chainer.cuda.to_gpu(img)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                # pred = self.model.predictor(chainer.Variable(img))
                data = self.model.predictor.extract(chainer.Variable(img), div_layer)
                if self.gpu >= 0:
                    data = chainer.cuda.to_cpu(data.data)
                else:
                    data = data.data
                return data


    def resume(self, x, div_layer):
        if self.gpu >= 0:
            x = chainer.cuda.to_gpu(x)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                data = self.model.predictor.resume(x, div_layer)
                if self.gpu >= 0:
                    data = chainer.cuda.to_cpu(data.data)
                else:
                    data = data.data
                return data

def main():
    P = predictor("model_iter_232600",0)
    img = cv2.imread("n03710193_4347.JPEG")
    times=[]
    for n in range(10):
        s = time.time()
        _a = P.extract(img, 'conv_bn')
        _b = P.resume(_a, 'conv_bn')
        times.append((time.time() - s) * 1000)
    print sum(times)/10

    idx = np.argmax(_b)
    with open("imagenet_class_index.json", "r") as f:
        class_idx = json.load(f)

    print("num:{} en:{} ja:{}".format(class_idx[idx]["num"], class_idx[idx]["en"], class_idx[idx]["ja"].encode('utf-8')))


if __name__ == '__main__':
    main()
