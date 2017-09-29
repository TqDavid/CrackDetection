# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:46:30 2017

@author: Jacky
"""

from PIL import Image
from IPython.display import display
import os
import h5py
import numpy as np
from scipy import ndimage

# In[0]: load images from folder
folder = '.\pydatabase\image'

image_width = 320  # Pixel width and height.
image_height = 480
num_channels = 3
pixel_depth = 255.0  # Number of levels per pixel.
half_block = 15
                   
def load(folder):
  """Load the training images data."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_width, image_height, num_channels),
                         dtype=np.float32)
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
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]   
  print('Full dataset tensor:', dataset.shape)
  return dataset

dataset = load(folder)

GT = h5py.File("./GT.mat")
crackGT = GT["GT"][:]
datalabel = np.ndarray(shape=(118, image_width, image_height))
for i in range(118):
    datalabel[i, :, :] = np.transpose(crackGT[:, :, i])
im = Image.fromarray(dataset[0, :, :, :].astype(np.uint8))
label = Image.fromarray(datalabel[0, :, :].astype(np.uint8))
display(im)
display(label)

# In[0]: depart
outpath = './pydatabase/test_pos'
num_block = 0
for num_image in range(100, 118):
    for x in range(0, image_width - half_block, 4):
        for y in range(0, image_height - half_block, 4):
            if datalabel[num_image, x, y] == 1 and x - half_block > 0 and y - half_block > 0:
                block = dataset[num_image, x-half_block:x+half_block, y-half_block:y+half_block, :]
                outfile = os.path.join(outpath, str(num_block)) + '.jpg'
                Image.fromarray(block.astype(np.uint8)).save(outfile)
                num_block += 1
    print("complete saving image %d" % num_image)


outpath = './pydatabase/test_neg'
num_block = 0
for num_image in range(100, 118):
    width_block = image_width // (2 * half_block)
    height_block = image_height // (2 * half_block)
    for x in range(width_block):
        for y in range(height_block):
            block_label = datalabel[num_image, x*2*half_block:(x+1)*2*half_block, y*2*half_block:(y+1)*2*half_block]
            if block_label.sum() == 0:
                block = dataset[num_image, x*2*half_block:(x+1)*2*half_block, y*2*half_block:(y+1)*2*half_block, :]
                outfile = os.path.join(outpath, str(num_block)) + '.jpg'
                Image.fromarray(block.astype(np.uint8)).save(outfile)
                num_block += 1
    print("complete saving image %d" % num_image)
    


#im = Image.open("G:/Crack Detection/project/Examples/001.jpg")
#GT = h5py.File("G:/Crack Detection/project/Examples/crackforest.mat")
#crackIm = GT["crackIm"]
#crackIm = np.array(crackIm)
#arrays = {}
#for k, v in GT.items():
#    arrays[k] = np.array(v)
#im = im.convert("L")
#test = array(im)
#imshow(im)
#display(test)
#im.show()