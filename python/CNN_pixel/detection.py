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
path = 'G:\Crack Detection\project\database\pydatabase'
folder = os.path.join(path, 'image')
block_size = 27
pixel_depth = 255.0
half_block = block_size // 2

graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph(path + '/CNN_cracks.meta')

# In[2]: Session
with tf.Session(graph=graph) as session:
    saver.restore(session, os.path.join(folder, "CNN_cracks"))
    predict = tf.get_collection('predict')[-1]
    data = tf.get_collection('data')[-1]
#    session.graph.finalize()
    for num_pic in range(1,119):
        
        start = time.time()
        
        # 读取图片
        sample =  str(num_pic) + '.jpg'
        path = os.path.join(folder, sample)
        img = cv2.imread(path)
        pic = (img - pixel_depth / 2) / pixel_depth
        
        # 图片高度、宽度读取
        width = pic.shape[0]
        height = pic.shape[1]
        width_block = width // block_size
        height_block = height // block_size
        
        # 单个样本验证
        x = 0
        y = 0
        pic_pad = np.zeros([width+block_size-1, height+block_size-1, 3])
        output = np.zeros([width, height])
        for i in range(3):
            pic_pad[:, :, i] = np.pad(pic[:, :, i], half_block, 'constant', constant_values=0)
        for x in range(width):
            for y in range(height):
                block = pic_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1, :]
                block = np.reshape(block, [1, block_size, block_size, 3])
                prediction = session.run(predict, feed_dict={data: block})
                if prediction == 1:
                    output[x,y] = 255
        result = Image.fromarray(output.astype(np.uint8)).save(path + '/localization/' + str(num_pic)+ '.png')
#       display(result)      
        print(time.time() - start)

session.close()
del session