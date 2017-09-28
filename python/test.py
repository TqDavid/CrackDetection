# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:57:36 2017

@author: Jacky
"""

import cv2
from PIL import Image
import numpy as np
from IPython.display import display

sample = './pydatabase/groundtruth/42.png'
img = cv2.imread(sample)
#red = img[:,:,2]
display(Image.fromarray(img))
#label = red > 240
#label = np.multiply(label, 1) * 255
#display(Image.fromarray(label.astype(np.uint8)))