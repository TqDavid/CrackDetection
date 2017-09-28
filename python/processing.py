# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 08:36:32 2017

@author: Jacky
"""
# In[0]: 模块导入
import tensorflow as tf
import os
#from IPython.display import display, Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# In[1]: 初始化处理
# 基本参数设置
folder = 'G:\Crack Detection\project'
sample = '002.jpg'
path = os.path.join(folder, sample)
block_size = 30
slide = 15
pixel_depth = 255.0

# 读取图片
img = cv2.imread(sample)
#cv2.imshow("origin", img)
#cv2.waitKey(0)
pic = (img - pixel_depth / 2) / pixel_depth

# 图片高度、宽度读取
width = pic.shape[0]
height = pic.shape[1]
width_block = width // block_size
height_block = height // block_size

# 定义输出矩阵，记录处理结果
output = np.zeros((width, height))

# In[2]: Session
with tf.Session() as session:
    saver = tf.train.import_meta_graph('./CNN_cracks.meta')
    saver.restore(session, "./CNN_cracks")
    predict = tf.get_collection('predict')[-1]
    data = tf.get_collection('data')[-1]
    x = 0
    y = 0
    block_num = 0
    while x + block_size < width:
        while y + block_size < height:
            block = pic[x:x+block_size, y:y+block_size, :]
            prediction = tf.argmax(session.run(predict, feed_dict={data: np.reshape(block, [1, block_size, block_size, 3])}), -1)
            if prediction.eval() == 0:
                blocktemp = img[x:x+block_size, y:y+block_size, :]                               # 存储对应的block
                imblock = cv2.cvtColor(blocktemp, cv2.COLOR_RGB2GRAY)                           # RGB转灰度
                blockout = imblock < 0.72 * cv2.blur(imblock, (15,15))
                blockout = np.multiply(blockout, 1)
                block_num += 1
                output[x:x+block_size, y:y+block_size] = blockout
            y += slide
        x += slide
        y = 0
    x = width
    while y + block_size < height:
        block = pic[x-block_size:x, y:y+block_size, :]
        prediction = tf.argmax(session.run(predict, feed_dict={data: np.reshape(block, [1, block_size, block_size, 3])}), -1)
        if prediction.eval() == 0:
            blocktemp = img[x-block_size:x, y:y+block_size, :]
            imblock = cv2.cvtColor(blocktemp, cv2.COLOR_RGB2GRAY)
            blockout = imblock < 0.72 * cv2.blur(imblock, (15,15))
            blockout = np.multiply(blockout, 1)
            output[x-block_size:x, y:y+block_size] = blockout
        y += slide
    y = height
    x = 0
    while x + block_size < width:
        block = pic[x:x+block_size, y-block_size:y, :]
        prediction = tf.argmax(session.run(predict, feed_dict={data: np.reshape(block, [1, block_size, block_size, 3])}), -1)
        if prediction.eval() == 0:
            blocktemp = img[x:x+block_size, y-block_size:y, :]
            imblock = cv2.cvtColor(blocktemp, cv2.COLOR_RGB2GRAY)
            blockout = imblock < 0.72 * cv2.blur(imblock, (15,15))
            blockout = np.multiply(blockout, 1)
            output[x:x+block_size, y-block_size:y] = blockout
        x += slide
    output *= 255
    plt.imshow(output,cmap='Greys_r')
    plt.axis("off")
#    result = Image.fromarray(output.astype(np.uint8))
#    display(result)

session.close()
del session