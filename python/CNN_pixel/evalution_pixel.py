# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:23:54 2017

@author: Ferris
"""

import cv2
import numpy as np
#from PIL import Image
#from IPython.display import display

#path = '../pydatabase'
path = "G:/Crack Detection/project/database/pydatabase"
image_path = path + '/localization_pixel/'
label_path = path + '/groundtruth/'

precision = list()
recall = list()
F1 = list()

for i in range(1,119):
    image= cv2.imread(image_path + str(i) + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    label = cv2.imread(label_path + str(i) + '.png')
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) // 255
    result = image > 0
    result = np.multiply(result, 1)
    TP = np.sum(result & label)
    precision.append(TP/np.sum(result))
    recall.append(TP/np.sum(label))
    F1.append(2 * precision[i - 1] * recall[i - 1] / (precision[i - 1] + recall[i - 1]))
    print("image number: ", i)
    print("precison = ", precision[i - 1])
    print("recall = ", recall[i - 1])
    print("F1 = ", F1[i - 1])
    
print("final precision: ", sum(precision)/len(precision))
print("final recall: ", sum(recall)/len(recall))
print("final F1: ", sum(F1)/len(F1))

print("max precision: ", max(precision))
print("max recall: ", max(recall))
print("max F1: ", max(F1))
