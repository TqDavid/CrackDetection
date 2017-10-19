# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:09:12 2017

@author: Jacky
"""
import numpy as np
import os
from scipy import ndimage

path = "G:/Crack Detection/project/database/pydatabase/image/"

# variables
image_width = 320  # Pixel width and height.
image_height = 480
num_channels = 3
pixel_depth = 255.0  # Number of levels per pixel.
half_block = 15

# definition
def load_image(folder):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_width, image_height, num_channels), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float))
            if image_data.shape != (image_width, image_height, num_channels):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e)
    dataset = dataset[0:num_images, :, :]
#    print('Full dataset tensor:', dataset.shape)
    return dataset

def load_GT(folder):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_width, image_height), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float))
            if image_data.shape != (image_width, image_height):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e)
    dataset = dataset[0:num_images, :, :]
#    print('Full dataset tensor:', dataset.shape)
    return dataset

