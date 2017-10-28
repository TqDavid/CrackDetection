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
    
print("final precision: ", sum(precision)/len(precision))
print("final recall: ", sum(recall)/len(recall))
print("final F1: ", sum(F1)/len(F1))

print("max precision: ", max(precision))
print("max recall: ", max(recall))
print("max F1: ", max(F1))
