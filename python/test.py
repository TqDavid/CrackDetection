# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:57:36 2017

@author: Jacky
"""
# In[0]:import
import cv2
from PIL import Image
import numpy as np
from IPython.display import display
import tensorflow as tf

# In[1]:sample
sample = 'G:\Crack Detection\project\database\pydatabase\image/001.jpg'
gt = "G:\Crack Detection\project\database\pydatabase\groundtruth/1.png"
img = cv2.imread(sample)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
label = cv2.imread(gt)
label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)

image_width = 320
image_height = 480
half_block = 16
pixel_depth = 255.0

num_block = 0
for x in range(0, image_width - half_block, 4):
    if num_block >= 21:
        break   
    for y in range(0, image_height - half_block, 4):
        if label[x, y] != 0 and x - half_block > 0 and y - half_block > 0:
            if num_block == 20:
                block = img_gray[x-half_block:x+half_block, y-half_block:y+half_block]
                block_gt = label[x-half_block:x+half_block, y-half_block:y+half_block]
                num_block += 1
                break
            else:
                num_block += 1
display(Image.fromarray(block.astype(np.uint8)))
block = (block - pixel_depth) / pixel_depth

# In[2]:Autoencoder
#Xaiver初始化器，作用是权重初始化，满足均值0和方差2/(in+out)，这里创建的是标准的均匀分布
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),minval = low, maxval = high, dtype = tf.float32)

class Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype = tf.float32))
        return all_weights
    
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})
        
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X})
        
    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
        
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X})
        
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
        
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

training_epochs = 20
auto = Autoencoder(n_input=32*32, n_hidden=20)
    
for epoch in range(training_epochs):
    cost = auto
    