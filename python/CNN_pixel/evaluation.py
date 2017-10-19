# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:23:54 2017

@author: Ferris
"""

import cv2
import numpy as np
#from PIL import Image
#from IPython.display import display

path = 'G:\Crack Detection\project\database\pydatabase'
image_path = path + '/localization/'
label_path = path + '/groundtruth/'

precision = list()
recall = list()
F1 = list()

for i in range(1,119):
    image= cv2.imread(image_path + str(i) + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    label = cv2.imread(label_path + str(i) + '.png')
    label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY) // 255
    result = image > 0
    result = np.multiply(result, 1)
    TP = np.sum(result & label)
    FP = np.sum(result & (~label))
    FN = np.sum((~result) & label)
    precision.append(TP/(TP + FP))
    recall.append(TP/(TP + FN))
    F1.append(2 * precision[i - 1] * recall[i - 1] / (precision[i - 1] + recall[i - 1]))
    print("precison = ", precision[i - 1])
    print("recall = ", recall[i - 1])
    print("F1 = ", F1[i - 1])
    
print("final precision: ", sum(precision)/len(precision))
print("final recall: ", sum(recall)/len(recall))
print("final F1: ", sum(F1)/len(F1))