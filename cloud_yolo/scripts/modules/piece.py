#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
いろんな部品をまとめておく
"""
import cv2
import numpy as np


class GammaCorrect:
    """
    ガンマ補正を行うクラス
    """
    def __init__(self, gamma):
        self.lookUpTable = np.zeros((256, 1), dtype='uint8')

        for i in range(256):
            self.lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    def __call__(self, img):
        img = cv2.LUT(img, self.lookUpTable)
        return img

