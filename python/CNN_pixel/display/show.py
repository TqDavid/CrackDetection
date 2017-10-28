# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:03:11 2017

@author: Jacky
"""

import numpy as np
import cv2
from IPython.display import display
from PIL import Image

def pixeldup(image, dupli):
    width = dupli * image.shape[0]
    height = dupli * image.shape[1]
    display = np.ones([width, height, 3])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for i in range(3):
                display[x*dupli : (x+1)*dupli, y*dupli : (y+1)*dupli, i] = image[x,y,i] * display[x*dupli : (x+1)*dupli, y*dupli : (y+1)*dupli, i]
    return display

sample = "G:/Crack Detection/project/database/pydatabase/example/pos/1.jpg"
gt = "G:/Crack Detection/project/CrackDetection/python/CNN_pixel/display/label.png"
image = cv2.imread(sample)
label = cv2.imread(gt)
display(Image.fromarray(image.astype(np.uint8)))
edge = label.shape[0]
center = edge // 2
show = pixeldup(image, 8)
output = label[center - 2:center + 3, center - 2:center + 3]
print(output.shape)
output = pixeldup(output, 8)
display(Image.fromarray(output.astype(np.uint8)))
#center = show.shape[0] // 2
#show[center - 8:center + 12, center - 8, :] = 0
#show[center - 8:center + 12, center + 12, :] = 0
#show[center - 8, center - 12:center + 12, :] = 0
#show[center + 12, center - 8:center + 16, :] = 0
display(Image.fromarray(show.astype(np.uint8)))
