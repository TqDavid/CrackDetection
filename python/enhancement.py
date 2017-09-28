# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 22:17:17 2017

@author: Jacky
"""

import tensorflow as tf

target = '.\pydatabase\image\0.jpg'
reader = tf.WholeFileReader()
key, value = reader.read(tf.train.string_input_producer([target]))
image0 = tf.image.decode_jpeg(value)