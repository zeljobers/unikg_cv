#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:48:17 2024

@author: zorin
# https://learnopencv.com/contour-detection-using-opencv-python-c/
"""
import cv2
 
# read the image
image = cv2.imread('./1.jpg')
image_complex = cv2.imread('./2.jpg')
# convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_gray_complex = cv2.cvtColor(image_complex, cv2.COLOR_BGR2GRAY)
# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
ret_complex, thresh_complex = cv2.threshold(img_gray_complex, 115, 255, cv2.THRESH_BINARY)
# visualize the binary image

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# detect the contours on the binary image using cv2.ChAIN_APPROX_SIMPLE
contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours1_complex, hierarchy1_complex = cv2.findContours(thresh_complex, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# print(hierarchy1)
# draw contours on the original image for `CHAIN_APPROX_SIMPLE`
image_copy1 = image.copy()
# draw contours on the original image for `CHAIN_APPROX_SIMPLE`
image_copy1_complex = image_complex.copy()
cv2.drawContours(image_copy1, contours1, -1, (0, 255, 0), 1, cv2.LINE_AA)
cv2.drawContours(image_copy1_complex, contours1_complex, -1, (0, 255, 0), 1, cv2.LINE_AA)
# # see the results
cv2.imshow('Simple approximation', image_copy1)
# cv2.imshow('Binary image', thresh)
cv2.imshow('No approximation', image_copy1_complex)
# cv2.imshow('Binary image', thresh_complex)
cv2.waitKey(0)
# cv2.imwrite('image_thres1.jpg', thresh)
cv2.imwrite('contours_simple_image1.jpg', image_copy1)
# cv2.imwrite('image_thres1_complex.jpg', thresh_complex)
cv2.imwrite('contours_image1_complex.jpg', image_copy1_complex)
cv2.destroyAllWindows()