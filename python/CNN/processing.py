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
folder = 'G:\Crack Detection\project\database\pydatabase/image'
sample = '001.jpg'
path = os.path.join(folder, sample)
block_size = 30
slide = 15
pixel_depth = 255.0

## 读取图片
#img = cv2.imread(sample)
#img_pos = img.copy()
##cv2.imshow("origin", img)
##cv2.waitKey(0)
#pic = (img - pixel_depth / 2) / pixel_depth
#
## 图片高度、宽度读取
#width = pic.shape[0]
#height = pic.shape[1]
#width_block = width // block_size
#height_block = height // block_size

# 定义输出矩阵，记录处理结果
#output = np.zeros((width, height))
graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph('G:\Crack Detection\project\database\pydatabase/CNN_cracks.meta')

# In[2]: Session
with tf.Session(graph=graph) as session:
#    saver = tf.train.import_meta_graph('./CNN_cracks.meta')
#    saver.restore(session, "./CNN_cracks")
#    predict = tf.get_collection('predict')[-1]
#    data = tf.get_collection('data')[-1]
#    prediction = tf.Variable()
    saver.restore(session, "G:\Crack Detection\project\database\pydatabase/CNN_cracks")
    predict = tf.get_collection('predict')[-1]
    data = tf.get_collection('data')[-1]
#    session.graph.finalize()
#    for num_pic in range(1,119):
    start = time.time()
    # 读取图片
#        sample =  str(num_pic) + '.jpg'
#        path = os.path.join(folder, sample)
    img = cv2.imread(path)
    img_pos = img.copy()
    pic = (img - pixel_depth / 2) / pixel_depth
    
    # 图片高度、宽度读取
    width = pic.shape[0]
    height = pic.shape[1]
    width_block = width // block_size
    height_block = height // block_size
    
    # 初始化遍历
    x = 0
    y = 0
    block_num = 0
    while x + block_size < width:
        while y + block_size < height:
            block_ = pic[x:x+block_size, y:y+block_size, :]
            block = np.reshape(block_, [1, block_size, block_size, 3])
            prediction = session.run(predict, feed_dict={data: block})
            if prediction == 1:
                img_pos[x:x+block_size, y:y+block_size, :] = 0
#                blocktemp = img[x:x+block_size, y:y+block_size, :]                              # 存储对应的block
#                imblock = cv2.cvtColor(blocktemp, cv2.COLOR_RGB2GRAY)                           # RGB转灰度
#                blockout = imblock < 0.72 * cv2.blur(imblock, (19,19))
#                blockout = np.multiply(blockout, 1)
#                    block_num += 1
#                output[x:x+block_size, y:y+block_size] = blockout
            y += slide
        x += slide
        y = 0
        
    # 边缘遍历
    x = width
    while y + block_size < height:
        block_ = pic[x-block_size:x, y:y+block_size, :]
        block = np.reshape(block_, [1, block_size, block_size, 3])
        prediction = session.run(predict, feed_dict={data: block})
        if prediction == 1:
            img_pos[x-block_size:x, y:y+block_size, :] = 0
#            blocktemp = img[x-block_size:x, y:y+block_size, :]
#            imblock = cv2.cvtColor(blocktemp, cv2.COLOR_RGB2GRAY)
#            blockout = imblock < 0.72 * cv2.blur(imblock, (19,19))
#            blockout = np.multiply(blockout, 1)
#            output[x-block_size:x, y:y+block_size] = blockout
        y += slide
    y = height
    x = 0
    while x + block_size < width:
        block = pic[x:x+block_size, y-block_size:y, :]
        block = np.reshape(block, [1, block_size, block_size, 3])
        prediction = session.run(predict, feed_dict={data: block})
        if prediction == 1:
            img_pos[x:x+block_size, y-block_size:y, :] = 0
#            blocktemp = img[x:x+block_size, y-block_size:y, :]
#            imblock = cv2.cvtColor(blocktemp, cv2.COLOR_RGB2GRAY)
#            blockout = imblock < 0.72 * cv2.blur(imblock, (19,19))
#            blockout = np.multiply(blockout, 1)
#            output[x:x+block_size, y-block_size:y] = blockout
        x += slide
            
    img_neg = img - img_pos
    result = Image.fromarray(img_neg.astype(np.uint8))#.save('./localization/'+str(num_pic)+'.jpg')
    display(result)
    print(time.time() - start)

#session.close()
#del session