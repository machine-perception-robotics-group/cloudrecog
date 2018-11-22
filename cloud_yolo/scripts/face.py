#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np

class net(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(net, self).__init__(
            conv1=L.Convolution2D(1, 16, 3),
            conv2=L.Convolution2D(8, 16, 3),
            conv3=L.Convolution2D(8, 32, 3),
            conv4=L.Convolution2D(16, 32, 3),
            conv5=L.Convolution2D(16, 64, 3),
            conv6=L.Convolution2D(32, 64, 3),
            conv7=L.Convolution2D(32, 128, 3),
            conv8=L.Convolution2D(64, 128, 2),
            conv9=L.Convolution2D(64, 256, 2),

            fc1    = L.Linear(512, 200),
            fpos   = L.Linear(200, 10),
            gender = L.Linear(200,2),
            age    = L.Linear(200,1),
            race   = L.Linear(200,3),
            smile  = L.Linear(200,1)
        )
        self.train = False

    def _normalize(self,x):
        diff_x = x.data - np.mean(x.data)
        print x.data
        print np.mean(x.data)
        print diff_x
        squared = diff_x ** 2
        stddev = np.sqrt(np.mean(squared))
        return Variable(diff_x / stddev)

    def __call__(self, x, t):
        '''
        #assert self. < self.ep ,"StartPoint or EndPoint ERROR!"
        h = self._normalize(self.conv1(x))
        h = F.maxout(h,2)
        #np.savetxt('hoge.txt',h.data,delimiter=',')
        h = self._normalize(self.conv2(h))
        h = F.maxout(h,2)
        h = F.max_pooling_2d(h,2)

        h = self._normalize(self.conv3(h))
        h = F.maxout(h,2)

        h = self._normalize(self.conv4(h))
        h = F.maxout(h,2)
        h = F.max_pooling_2d(h,2)

        h = self._normalize(self.conv5(h))
        h = F.maxout(h,2)

        h = self._normalize(self.conv6(h))
        h = F.maxout(h,2)
        h = F.max_pooling_2d(h,2)

        h = self._normalize(self.conv7(h))
        h = F.maxout(h,2)

        h = self._normalize(self.conv8(h))
        h = F.maxout(h,2)
        h = F.max_pooling_2d(h,2)

        h = self._normalize(self.conv9(h))
        h = F.maxout(h,2)

        h = self.fc1(h)
        h = F.dropout(F.sigmoid(h), train=self.train)
        h_fpos = F.sigmoid(self.fpos(h))
        h_gender = F.softmax(self.gender(h))
        h_age = F.sigmoid(self.age(h))
        h_race = F.softmax(self.race(h))
        h_smile = F.sigmoid(self.smile(h))


        self.pred = [h_fpos, h_gender, h_age, h_race, h_smile]
        return self.pred
        '''
        h = self._normalize(x)
        h = self.conv1(h)
        h = F.maxout(h, 2)
        h = self._normalize(h)

        h = self.conv2(h)
        h = F.maxout(h, 2)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self._normalize(h)

        h = self.conv3(h)
        h = F.maxout(h, 2)
        h = self._normalize(h)

        h = self.conv4(h)
        h = F.maxout(h, 2)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self._normalize(h)

        h = self.conv5(h)
        h = F.maxout(h, 2)
        h = self._normalize(h)

        h = self.conv6(h)
        h = F.maxout(h, 2)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self._normalize(h)

        h = self.conv7(h)
        h = F.maxout(h, 2)
        h = self._normalize(h)

        h = self.conv8(h)
        h = F.maxout(h, 2)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self._normalize(h)

        h = self.conv9(h)
        h = F.maxout(h, 2)
        h = self._normalize(h)

        h = 0.5 * h # Theano の Dropout は評価時に 0.5 を乗算するため (chainer は乗算しない)
        h = self.fc1(h)
        h = F.sigmoid(h)
        h = self._normalize(h)

        h_fpos = F.sigmoid(self.fpos(h))
        h_gender = F.softmax(self.gender(h))
        h_age = F.sigmoid(self.age(h))
        h_race = F.softmax(self.race(h))
        h_smile = F.sigmoid(self.smile(h))


        self.pred = [h_fpos, h_gender, h_age, h_race, h_smile]
        return self.pred
