#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:04:48 2024

@author: zorin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt 



image = cv2.imread('sp4.jpg') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
plt.imshow(gray, cmap='gray') 



blur = cv2.GaussianBlur(gray, (11, 11), 0) 
plt.imshow(blur, cmap='gray') 


canny = cv2.Canny(blur, 30, 150, 3) 
plt.imshow(canny, cmap='gray') 


dilated = cv2.dilate(canny, (1, 1), iterations=0) 
plt.imshow(dilated, cmap='gray') 



(cnt, hierarchy) = cv2.findContours( 
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
cv2.drawContours(rgb, cnt, -1, (255, 0, 0), 2) 
  
plt.imshow(rgb) 


print("coins in the image : ", len(cnt)) 
