#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:25:37 2024

@author: zorin
"""

import numpy as np
import cv2
import os

# for dirname in os.listdir("images/"):

    # for filename in os.listdir("images/" + dirname + "/"):

        # Image read
img = cv2.imread("sp4.jpg", 0)

        # Denoising
denoisedImg = cv2.fastNlMeansDenoising(img);

        # Threshold (binary image)
        # thresh – threshold value.
        # maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
        # type – thresholding type
th, threshedImg = cv2.threshold(denoisedImg, 200, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU) # src, thresh, maxval, type

        # Perform morphological transformations using an erosion and dilation as basic operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)

        # Find and draw contours
contours, hierarchy = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
cv2.drawContours(contoursImg, contours, -1, (0,0,255), 3)

cv2.imwrite("_result.jpg", contoursImg)
textFile = open("results.txt","a")
textFile.write("_result.jpg" + " Dots number: {}".format(len(contours)) + "\n")
print("Dots number: {}".format(len(contours)))
textFile.close()