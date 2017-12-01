# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 08:36:32 2017

@author: Jacky
"""
# In[0]: 模块导入
import tensorflow as tf
import os
from IPython.display import display
import time
from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import cv2

# In[1]: 初始化处理
# 基本参数设置
#path = 'G:\Crack Detection\project\database\pydatabase'
#path = '../pydatabase'
forest = 1
if forest == 1:
    path = '../pydatabase/CFD'
    form = '.jpg'
else:
    path = '../pydatabase/AigleRN'
    form = '.png'
folder = os.path.join(path, 'test')    
block_size = 27
struct = 1
pixel_depth = 255.0
half_block = block_size // 2
half_struct = struct // 2

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph(path + '/CNN_cracks.meta')

# In[2]: Session
with tf.Session(graph=graph) as session:
    saver.restore(session, os.path.join(path, "CNN_cracks"))
    predict = tf.get_collection('predict')[-1]
    data = tf.get_collection('data')[-1]
#    session.graph.finalize()
    image_files = os.listdir(folder)
    for num_pic in range(1, len(image_files) + 1):
        
        start = time.time()
        
        # 读取图片
        sample = folder + '/' + str(num_pic) + form
        img = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2GRAY)
        pic = (img - pixel_depth / 2) / pixel_depth * 2
        
        # 图片高度、宽度读取
        width = pic.shape[0]
        height = pic.shape[1]
        width_block = width // block_size
        height_block = height // block_size
        
        # 单个样本验证
        x = 0
        y = 0
        output = np.zeros([width, height])
        outpad = np.pad(output, half_struct, 'symmetric')
        pic_pad = np.pad(pic, half_block, 'symmetric')
        for x in range(width):
            block_tensor = list()
            for y in range(height):
                block = pic_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1]
                block_tensor.append(block)
            block_tensor = np.array(block_tensor)
            block_tensor = np.reshape(block_tensor, [-1, block_size, block_size, 1])
            prediction = session.run(predict, feed_dict={data: block_tensor})
            for y in range(height):
                outpad[x:x + 2*half_struct + 1, y:y + 2*half_struct + 1] += np.reshape(prediction[y], [struct, struct])
        output = outpad[half_struct:width+half_struct, half_struct:height+half_struct]
        output = output / np.max(output) * 255
        result = Image.fromarray((output).astype(np.uint8)).save(path + '/localization_struct/' + str(num_pic)+ '.png')
#        display(result)      
        print(time.time() - start)

session.close()
del session
