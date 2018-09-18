#!/usr/bin/env python
# -*- Coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
import collections

class ConvBN(chainer.Chain):
    def __init__(self, inp, oup, stride):
        super(ConvBN, self).__init__()
        with self.init_scope():
            self.conv=L.Convolution2D(inp, oup, 3, stride=stride, pad=1, nobias=True)
            self.bn=L.BatchNormalization(oup)
            # self.scale=L.Scale(1, oup, bias_term=True)

    def __call__(self, x):
        # h = F.relu(self.scale(self.bn(self.conv(x))))
        h = F.relu(self.bn(self.conv(x)))
        return h


class ConvDW(chainer.Chain):
    def __init__(self, inp, oup, stride):
        super(ConvDW, self).__init__()
        with self.init_scope():
            self.conv_dw=L.DepthwiseConvolution2D(inp, 1, 3, stride=stride, pad=1, nobias=True)
            self.bn_dw=L.BatchNormalization(inp)
            self.conv_sep=L.Convolution2D(inp, oup, 1, stride=1, pad=0, nobias=True)
            self.bn_sep=L.BatchNormalization(oup)

    def __call__(self, x):
        h = F.relu(self.bn_dw(self.conv_dw(x)))
        h = F.relu(self.bn_sep(self.conv_sep(h)))
        return h


# Network definition
class MobileNet(chainer.Chain):
    insize=224

    def __init__(self):
        super(MobileNet, self).__init__()
        with self.init_scope():
            self.conv_bn = ConvBN(3, 32, 2)
            self.conv_ds_2 = ConvDW(32, 64, 1)
            self.conv_ds_3 = ConvDW(64, 128, 2)
            self.conv_ds_4 = ConvDW(128, 128, 1)
            self.conv_ds_5 = ConvDW(128, 256, 2)
            self.conv_ds_6 = ConvDW(256, 256, 1)
            self.conv_ds_7 = ConvDW(256, 512, 2)
            self.conv_ds_8 = ConvDW(512, 512, 1)
            self.conv_ds_9 = ConvDW(512, 512, 1)
            self.conv_ds_10 = ConvDW(512, 512, 1)
            self.conv_ds_11 = ConvDW(512, 512, 1)
            self.conv_ds_12 = ConvDW(512, 512, 1)
            self.conv_ds_13 = ConvDW(512, 1024, 2)
            self.conv_ds_14 = ConvDW(1024, 1024, 1)
            self.fc7 = L.Linear(1024, 1000)

    def __call__(self, x):
        h = self.conv_bn(x)
        h = self.conv_ds_2(h)
        h = self.conv_ds_3(h)
        h = self.conv_ds_4(h)
        h = self.conv_ds_5(h)
        h = self.conv_ds_6(h)
        h = self.conv_ds_7(h)
        h = self.conv_ds_8(h)
        h = self.conv_ds_9(h)
        h = self.conv_ds_10(h)
        h = self.conv_ds_11(h)
        h = self.conv_ds_12(h)
        h = self.conv_ds_13(h)
        h = self.conv_ds_14(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        # x = F.average(x, axis=(2, 3),keepdims=True)
        h = self.fc7(h)
        return h

    def _global_average_pooling_2d(self, x):
        n, channel, rows, cols = x.data.shape
        h = F.average_pooling_2d(x, (rows, cols), stride=1)
        h = F.reshape(h, (n, channel))
        return h

    def functions(self):
        return collections.OrderedDict([
            ('conv_bn', [self.conv_bn]),
            ('conv_ds_2', [self.conv_ds_2]),
            ('conv_ds_3', [self.conv_ds_3]),
            ('conv_ds_4', [self.conv_ds_4]),
            ('conv_ds_5', [self.conv_ds_5]),
            ('conv_ds_6', [self.conv_ds_6]),
            ('conv_ds_7', [self.conv_ds_7]),
            ('conv_ds_8', [self.conv_ds_8]),
            ('conv_ds_9', [self.conv_ds_9]),
            ('conv_ds_10', [self.conv_ds_10]),
            ('conv_ds_11', [self.conv_ds_11]),
            ('conv_ds_12', [self.conv_ds_12]),
            ('conv_ds_13', [self.conv_ds_13]),
            ('conv_ds_14', [self.conv_ds_14]),
            ('pool', [self._global_average_pooling_2d]),
            ('fc7', [self.fc7])
        ])

    def available_layers(self):
        return list(self.functions.keys())

    def extract(self, x, div_layer):
        functions = self.functions()
        if (div_layer not in functions):
            raise ValueError("{}(your given) not in {}".format(div_layer, functions.keys()))

        # self.functions.keys()
        for key, funcs in functions.items():
            # print("{}".format(key))
            for func in funcs:
                x = func(x)
            if key == div_layer:
                return x
    def resume(self, x, layer):
        """resume(self, x, layer)

        :param x: (~chainer.Variable): Input variable. It should be prepared by ``prepare`` function.
        :param layer: (str) The start layer you want resume.
        :return: (~chainer.Variable): output feature
        """
        h = x

        layer_idx = [s[0] for s in self.functions().items()].index(layer)
        # print self.functions().items()
        for key, funcs in self.functions().items()[(layer_idx+1):]:
            for func in funcs:
                h = func(h)
                # print("{}".format(key))

        return h




def main():
    # for testing
    t = MobileNet()
    t.extract(None, 'conv_ds_3')
    t.resume(None, 'conv_ds_3')


if __name__ == '__main__':
    main()


