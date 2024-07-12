#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:17:48 2024

@author: zorin
"""

import cv2
import numpy as np

img = cv2.imread("rgb.png")
b,g,r = cv2.split(img)
cv2.imshow("Red", r)
cv2.imshow("Green", g)
cv2.imshow("Blue", b)
cv2.waitKey(0)

zeros = np.zeros(img.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, r]))
cv2.imshow("Red", cv2.merge([zeros, g, zeros]))
cv2.imshow("Red", cv2.merge([b, zeros, zeros]))

cv2.waitKey(0)

cv2.destroyAllWindows()