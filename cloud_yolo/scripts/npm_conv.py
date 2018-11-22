#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
================================
numpyとFloat32MultiArrayの相互変換ツール
__author__ = 'inoko'
__copyright__ = "Copyright (C) 2016 inoko. All Rights Reserved."
__license__ = 'MIT'
"""

import numpy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension



def np_to_Float32MultiArray(data):
    """
    任意の次元，サイズのnumpy配列から，std_msgs/Float32MultiArray形式に変換する関数
    入力するndarrayはnumpy.float32を想定
    """
    data_shape = list(data.shape)
    data_shape.reverse()
    mat = Float32MultiArray()
    for n in range(len(data_shape)):
        mat.layout.dim.append(MultiArrayDimension())
        mat.layout.dim[n].label = "dim" + str(n)
        mat.layout.dim[n].size = data_shape[n]
        mat.layout.dim[n].stride = reduce(lambda x,y:x*y,data_shape[n:len(data_shape)])
    mat.layout.data_offset = 0
    mat.data = [0]*mat.layout.dim[0].stride
    mat.data = list(data.reshape((1,int(mat.layout.dim[0].stride)))[0,])
    return mat
def Float32MultiArray_to_np(intake):
    """
    任意の次元，サイズのstd_msgs/Float32MultiArray形式からnumpy形式に変換する関数
    出力するndarrayはnumpy.float32
    """
    l=[]
    for i in intake.layout.dim:
        l.append(i.size)
    l.reverse()
    np = numpy.array(intake.data,dtype=numpy.float32)
    return np.reshape(l)
