# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:23:47 2017

@author: Jacky
"""
import os
import numpy as np
from scipy import ndimage
from PIL import Image
from IPython.display import display
from six.moves import cPickle as pickle

# In[0]: load images and groundtruth from folder
path = '.\pydatabase'
#
#image_folder = '.\pydatabase\image'
#GT_folder = '.\pydatabase\groundtruth'

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

## load images and groundtruth    
#images = load_image(image_folder)
#groundtruth = load_GT(GT_folder)
#
## display examples
#im = Image.fromarray(images[0, :, :, :].astype(np.uint8))
#label = Image.fromarray(groundtruth[0, :, :].astype(np.uint8))
#display(im)
#display(label)

# In[1]: depart
def depart_pos(image_folder):
    GT_folder = image_folder + '_GT'
    dataset = list()
    images = load_image(image_folder)
    labels = load_GT(GT_folder)
    num_block = 0
    for num_image in range(np.shape(images)[0]):
        for x in range(0, image_width - half_block, 4):
            for y in range(0, image_height - half_block, 4):
                if labels[num_image, x, y] != 0 and x - half_block > 0 and y - half_block > 0:
                    block = images[num_image, x-half_block:x+half_block, y-half_block:y+half_block, :]
                    dataset.append(block)
                    num_block += 1
#        print("Complete departing image %d" % num_image)
    label = list(np.ones(len(dataset)))
    label = [int(i) for i in label]
    print("data positive: full dataset %d" % num_block)
    return dataset, label
    
def depart_neg(image_folder):
    GT_folder = image_folder + '_GT'
    dataset = list()
    images = load_image(image_folder)
    labels = load_GT(GT_folder)
    num_block = 0
    for num_image in range(np.shape(images)[0]):
        width_block = image_width // (2 * half_block)
        height_block = image_height // (2 * half_block)
        for x in range(width_block):
            for y in range(height_block):
                block_label = labels[num_image, x*2*half_block:(x+1)*2*half_block, y*2*half_block:(y+1)*2*half_block]
                if block_label.sum() == 0:
                    block = images[num_image, x*2*half_block:(x+1)*2*half_block, y*2*half_block:(y+1)*2*half_block, :]
                    dataset.append(block)
                    num_block += 1
#        print("Complete departing image %d" % num_image)
    label = list(np.zeros(len(dataset)))
    label = [int(i) for i in label]
    print("data negative: full dataset %d" % num_block)
    return dataset, label

test_datasets_pos, test_label_pos = depart_pos('./pydatabase/test')
test_datasets_neg, test_label_neg = depart_neg('./pydatabase/test')
valid_datasets_pos, valid_label_pos = depart_pos('./pydatabase/validation')
valid_datasets_neg, valid_label_neg = depart_neg('./pydatabase/validation')
train_datasets_pos, train_label_pos = depart_pos('./pydatabase/train')
train_datasets_neg, train_label_neg = depart_neg('./pydatabase/train')

train_dataset = np.array(train_datasets_pos + train_datasets_neg)
train_dataset = (train_dataset - pixel_depth / 2) / pixel_depth
train_labels = np.array(train_label_pos + train_label_neg)
valid_dataset = np.array(valid_datasets_pos + valid_datasets_neg)
valid_dataset = (valid_dataset - pixel_depth / 2) / pixel_depth
valid_labels = np.array(valid_label_pos + valid_label_neg)
test_dataset = np.array(test_datasets_pos + test_datasets_neg)
test_dataset = (test_dataset - pixel_depth / 2) / pixel_depth
test_labels = np.array(test_label_pos + test_label_neg)

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(path, 'cracks.pickle')

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
  
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
#im = Image.fromarray(test_datasets_pos[0].astype(np.uint8))
#label = Image.fromarray(test_datasets_neg[0].astype(np.uint8))
#display(im)
#display(label)