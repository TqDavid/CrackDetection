# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:23:54 2017

@author: Ferris
"""

import cv2
import numpy as np
from PIL import Image
from IPython.display import display
import os

path = '../pydatabase'
forest = 1
if forest == 1:
    image_path = path + '/CFD/localization_struct/'
    label_path = path + '/CFD/test_GT/'
else:
    image_path = path + '/AigleRN/localization_struct/'
    label_path = path + '/AigleRN/test_GT/'
pixel_depth = 255.0
thresh = 0.8

precision = list()
recall = list()
F1 = list()

image_files = os.listdir(image_path)
for i in range(1, len(image_files) + 1):
    image= cv2.imread(image_path + str(i) + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    label = cv2.imread(label_path + str(i) + '.png')
    label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY) // 255
    if forest == 0:
        label = 1 - label
    result = image > thresh * pixel_depth
    result = np.multiply(result, 1)
#    display(Image.fromarray((result * 255).astype(np.uint8)))
#    display(Image.fromarray((label * 255).astype(np.uint8)))
    width = image.shape[0]
    height = image.shape[1]
    TP_Pr = 0
    TP_Re = 0
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            if result[x,y] == 1 and np.sum(label[x-2:x+3,y-2:y+3]) > 0:
                TP_Pr += 1
            if label[x,y] == 1 and np.sum(result[x-2:x+3,y-2:y+3]) > 0:
                TP_Re += 1
    precision.append(TP_Pr/np.sum(result))
    recall.append(TP_Re/np.sum(label))
    F1.append(2 * precision[i - 1] * recall[i - 1] / (precision[i - 1] + recall[i - 1]))
    print("image number: ", i)
    print("precison = ", precision[i - 1])
    print("recall = ", recall[i - 1])
    print("F1 = ", F1[i - 1])

precision = np.array(precision)
recall = np.array(recall)
F1 = np.array(F1)
print("final precision: ", np.nansum(precision)/len(precision))
print("final recall: ", np.nansum(recall)/len(recall))
print("final F1: ", np.nansum(F1)/len(F1))

print("max precision: ", max(precision))
print("max recall: ", max(recall))
print("max F1: ", max(F1))
