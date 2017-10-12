# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:05:00 2017

@author: Jacky
"""
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import os
import cv2
from IPython.display import display
from PIL import Image

## 读取数据
path = "G:\Crack Detection\project\database\pydatabase"
pickle_file = os.path.join(path, 'cracks.pickle')

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
image_size = 30
num_labels = 2
num_channels = 3

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 32
patch_size = 3
depth = 32
num_hidden = 32

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  # 定义用于测试用的变量，并加入存储
  data = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels))
  tf.add_to_collection('data', data)
  
  # Variables，训练参数
  # 第一卷积层
  conv1_1_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
  conv1_1_biases = tf.Variable(tf.constant(0.0, shape=[depth]))
  conv1_2_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  conv1_2_biases = tf.Variable(tf.zeros([depth]))
  # 第二卷积层
  conv2_1_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  conv2_1_biases = tf.Variable(tf.constant(0.0, shape=[depth]))
  conv2_2_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  conv2_2_biases = tf.Variable(tf.zeros([depth]))
  # 第三卷积层
  conv3_1_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  conv3_1_biases = tf.Variable(tf.zeros([depth]))
  conv3_2_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  conv3_2_biases = tf.Variable(tf.zeros([depth]))
  # 第四全连接层
  fc4_weights = tf.Variable(tf.truncated_normal(
          [(image_size // 16) * (image_size // 16) * depth, num_hidden], stddev=0.1))
  fc4_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden]))
  # 第五全连接层
  fc5_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
  fc5_biases = tf.Variable(tf.constant(0.0, shape=[num_labels]))
  
  # 打印每一层的输出
  def print_activations(t):
      print(t.op.name, ' ', t.get_shape().as_list())
      
  # Model.
  def model(data):
      # 第一层卷积层，包含池化层
      with tf.name_scope('conv1') as scope:
          conv = tf.nn.conv2d(data, conv1_1_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv1_1_biases)
          conv = tf.nn.conv2d(activation, conv1_2_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv1_2_biases)
          pool = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name=scope)
          conv1 = tf.nn.dropout(pool, 1, name=scope)
          print_activations(conv1)
      # 第二层卷积层，包含池化层
      with tf.name_scope('conv2') as scope:
          conv = tf.nn.conv2d(conv1, conv2_1_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv2_1_biases)
          conv = tf.nn.conv2d(activation, conv2_2_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv2_2_biases)
          pool = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name=scope)
          conv2 = tf.nn.dropout(pool, 1, name=scope)
          print_activations(conv2)
      # 第三层卷积层，包含池化层
      with tf.name_scope('conv3') as scope:
          conv = tf.nn.conv2d(conv2, conv3_1_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv3_1_biases)
          conv = tf.nn.conv2d(activation, conv3_2_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv3_2_biases)
          pool = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name=scope)
          conv3 = tf.nn.dropout(pool, 0.5, name=scope)
          print_activations(conv3)
      # 第四层全连接层
      with tf.name_scope('fc4') as scope:
          shape = conv3.get_shape().as_list()
          reshape = tf.reshape(conv3, [shape[0], shape[1] * shape[2] * shape[3]])
          drop4 = tf.nn.relu(tf.matmul(reshape, fc4_weights) + fc4_biases)
          fc4 = tf.nn.dropout(drop4, 0.5, name=scope)
      # 第五层全连接层
      with tf.name_scope('fc5') as scope:
          fc5 = tf.matmul(fc4, fc5_weights) + fc5_biases
      return fc5

  def model_test(data):
      # 第一层卷积层，包含池化层
      with tf.name_scope('conv1') as scope:
          conv = tf.nn.conv2d(data, conv1_1_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv1_1_biases)
          conv = tf.nn.conv2d(activation, conv1_2_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv1_2_biases)
          pool = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name=scope)
          conv1 = tf.nn.dropout(pool, 1, name=scope)
      # 第二层卷积层，包含池化层
      with tf.name_scope('conv2') as scope:
          conv = tf.nn.conv2d(conv1, conv2_1_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv2_1_biases)
          conv = tf.nn.conv2d(activation, conv2_2_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv2_2_biases)
          pool = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name=scope)
          conv2 = tf.nn.dropout(pool, 1, name=scope)
      # 第三层卷积层，包含池化层
      with tf.name_scope('conv3') as scope:
          conv = tf.nn.conv2d(conv2, conv3_1_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv3_1_biases)
          conv = tf.nn.conv2d(activation, conv3_2_kernel, [1, 2, 2, 1], padding='SAME')
          activation = tf.nn.relu(conv + conv3_2_biases)
          pool = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name=scope)
          conv3 = tf.nn.dropout(pool, 1, name=scope)
      # 第四层全连接层
      with tf.name_scope('fc4') as scope:
          shape = conv3.get_shape().as_list()
          reshape = tf.reshape(conv3, [shape[0], shape[1] * shape[2] * shape[3]])
          drop4 = tf.nn.relu(tf.matmul(reshape, fc4_weights) + fc4_biases)
          fc4 = tf.nn.dropout(drop4, 1, name=scope)
      # 第五层全连接层
      with tf.name_scope('fc5') as scope:
          fc5 = tf.matmul(fc4, fc5_weights) + fc5_biases
      return fc5

  # Training computation.
  logits = model(tf_train_dataset)
  beta = 0.001
  l2_loss =  tf.nn.l2_loss(tf.concat(
          [tf.reshape(conv1_1_kernel, [-1]), tf.reshape(conv1_2_kernel, [-1]), 
           tf.reshape(conv2_1_kernel, [-1]), tf.reshape(conv2_2_kernel, [-1]),
           tf.reshape(conv3_1_kernel, [-1]), tf.reshape(conv3_2_kernel, [-1]),
           tf.reshape(fc4_weights, [-1]), tf.reshape(fc5_weights, [-1])], 0))
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + beta * l2_loss
    
  # Optimizer.
  optimizer = tf.train.AdadeltaOptimizer(2).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model_test(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model_test(tf_test_dataset))
  
#  predict = tf.nn.softmax(model_test(data))
  predict = tf.argmax(tf.nn.softmax(model_test(data)), -1)
  
  ## 保存模型，生成saver
  tf.add_to_collection('predict', predict)
  saver = tf.train.Saver(max_to_keep=1)
  
num_steps = 7001
train_accuracy = [0.0]
valid_accuracy = [0.0]

width = 320
height = 480

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    start_time = time.time()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            accuracy1 = accuracy(predictions, batch_labels)
            accuracy2 = accuracy(valid_prediction.eval(), valid_labels)
            train_accuracy.append(accuracy(predictions, batch_labels))
            valid_accuracy.append(accuracy(valid_prediction.eval(), valid_labels))
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    saver.save(session, "G:\Crack Detection\project\database\pydatabase\CNN_cracks")
    train_plot = tf.constant(train_accuracy).eval()
    valid_plot = tf.constant(valid_accuracy).eval()
    
    
    # 单个样本验证
    sample = "G:\Crack Detection\project\database\pydatabase\image/001.jpg"
    x = 0
    y = 0
    block_size = image_size
    slide = 15
    pixel_depth = 255.0
    pic = cv2.imread(sample)
    img_pos = pic.copy()
    pic = (pic - pixel_depth / 2) / pixel_depth
    while x + block_size < width:
        while y + block_size < height:
            block_ = pic[x:x+block_size, y:y+block_size, :]
            block = np.reshape(block_, [1, block_size, block_size, 3])
            prediction = session.run(predict, feed_dict={data: block})
            if prediction == 1:
                img_pos[x:x+block_size, y:y+block_size, :] = 0
            y += slide
        x += slide
        y = 0
    result = Image.fromarray(img_pos.astype(np.uint8))
    display(result)
    
    plotx = np.arange(0, num_steps, 100)
    plt.plot(plotx, 100 - train_plot[1:], color="#348ABD", label="training")
    plt.plot(plotx, 100 - valid_plot[1:], color="#A60628", label="validation")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Error rate")
    plt.title("Learning curve")
#    saver.restore(session, "G:\Crack Detection\project\database\pydatabase\CNN_cracks")
    print(time.time() - start_time)
#    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    
session.close()
del session

# In[]:画图比较学习率
#valid_accuracy3 = valid_accuracy.copy()
#plotx = np.arange(0, 10001, 100)
#plt.plot(plotx, 100 - np.array(valid_accuracy1[1:]), color="#348ABD", label=r"$\alpha = 1e-4$")
#plt.plot(plotx, 100 - np.array(valid_accuracy2[1:]), color="#A60628", label=r"$\alpha = 1e-3$")
#plt.plot(plotx, 100 - np.array(valid_accuracy3[1:]), color="#7A68A6", label=r"$\alpha = 1e-2$")
#plt.xlabel("Training steps")
#plt.ylabel("Error rate")
#plt.title("Learning rate")
#plt.legend()