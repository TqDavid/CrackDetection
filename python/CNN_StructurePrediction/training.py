# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:16:12 2017

由于代码放在服务器，python版本和tensorflow版本不一样，此代码为电脑本地兼容的版本

python: 3.5
tensoflow: 1.3

@author: Ferris
"""


# In[0]: import modules
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import os
import cv2
from IPython.display import display
from PIL import Image
#from scipy import ndimage

# In[0]: parameters

# 数据库路径
#path = 'G:/Crack Detection/project/database/pydatabase'
#sample = "G:/Crack Detection/project/database/pydatabase/image/1.jpg"
#path = 'E:/project/database/pydatabase'
#sample = "E:/project/database/pydatabase/image/001.jpg"
sample = "../pydatabase/1.jpg"

forest = 1
if forest == 1:
    path = '../pydatabase/CFD'
else:
    path = '../pydatabase/AigleRN'

# variable
num_channels = 1
image_size = 27
half_block = image_size // 2
pixel_depth = 255.0
struct = 5
half_struct = struct // 2
num_labels = struct * struct

batch_size = 256
patch_size = 3
depth_1 = 16
depth_2 = 32
num_hidden = 64


# In[1]: depart
    
# 打乱顺序
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

# 整理成可用于训练的格式
def reformat(dataset, labels):
    dataset = dataset.reshape(
            (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = labels.reshape(
            (-1, num_labels)).astype(np.float32)
    return dataset, labels    

# 分离样本
def depart(image_folder):
    GT_folder = image_folder + '_GT'
    datapos = list()
    dataneg = list()
    labelpos = list()
    labelneg = list()
    image_files = os.listdir(image_folder)
    num_images = 0
    pos_block = 0
    neg_block = 0
    for image in image_files:
        image_file = os.path.join(image_folder, image)
        GT = image[:-3] + "png"
        label_file = os.path.join(GT_folder, GT)
        try:
            image_data = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY)
#            display(Image.fromarray(image_data.astype(np.uint8)))
            label_data = cv2.cvtColor(cv2.imread(label_file), cv2.COLOR_BGR2GRAY) // 255
            image_width = image_data.shape[0]
            image_height = image_data.shape[1]
            if forest == 0:
                label_data = 1 - label_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e)
        label_pad = np.pad(label_data, half_struct, 'symmetric')
        image_pad = np.pad(image_data, half_block, 'symmetric')
        for x in range(0, image_width, 2):
            for y in range(0, image_height, 2):
                if label_data[x, y] != 0:
                    image_block = image_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1]
                    image_block = (image_block - pixel_depth / 2) / pixel_depth * 2
                    label_block = label_pad[x:x + 2*half_struct + 1, y:y + 2*half_struct + 1]
                    datapos.append(image_block)
                    labelpos.append(np.reshape(label_block, -1))
                    pos_block += 1
                    
        for x in range(0, image_width, 10):
            for y in range(0, image_height, 10):
                if label_data[x, y] == 0 and neg_block < 60*pos_block:
                    image_block = image_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1]
                    image_block = (image_block - pixel_depth / 2) / pixel_depth * 2
                    label_block = label_pad[x:x + 2*half_struct + 1, y:y + 2*half_struct + 1]                    
                    dataneg.append(image_block)
                    labelneg.append(np.reshape(label_block, -1))
                    neg_block += 1

#        for x in range(image_width-1, 0, -1):
#            for y in range(image_height-1, 0, -1):
#                if label_data[x, y] == 0 and neg_block < 50*pos_block:
#                    image_block = image_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1]
#                    image_block = (image_block - pixel_depth / 2) / pixel_depth
#                    label_block = label_pad[x:x + 2*half_struct + 1, y:y + 2*half_struct + 1]                    
#                    dataneg.append(image_block)
#                    labelneg.append(np.reshape(label_block, -1))
#                    neg_block += 1
    label = labelpos + labelneg
    dataset = datapos + dataneg
    del datapos, dataneg, labelpos, labelneg
    dataset1 = np.array(dataset)
    label1 = np.array(label)
    dataset2, label2 = randomize(dataset1, label1)
    dataset3, label3 = reformat(dataset2, label2)
    print("data positive: %d" % pos_block)
    print("data negative: %d" % neg_block)
    return dataset3, label3

valid_dataset, valid_label = depart(os.path.join(path, 'test'))
train_dataset, train_label = depart(os.path.join(path, 'train'))

print('Training set', train_dataset.shape, train_label.shape)
print('Validation set', valid_dataset.shape, valid_label.shape)

def accuracy(predictions, labels):
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    TP = np.nansum(predictions.astype(np.uint8) & labels.astype(np.uint8))
    precision = TP / np.nansum(predictions)
    recall = TP / np.nansum(labels)
    F1 = 2*precision*recall / (precision + recall)
#    batch = predictions.shape[0]
#    temp = (predictions[:] - labels[:])**2
#    loss = np.sum(np.reshape(temp, [batch, -1]), 1) / 2
#    loss = np.mean(loss)
    return 100*F1 #100*loss
#    compare = np.sum(predict & labels)
#    return (100.0 * compare / np.sum(predictions))

# In[2]: model   

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.placeholder(
            tf.float32, shape=(None, image_size, image_size, num_channels))
    tf_test_dataset = tf.placeholder(
            tf.float32, shape=(None, image_size, image_size, num_channels))
    
    # 定义用于测试用的变量，并加入存储
    data = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
    tf.add_to_collection('data', data)

    def xavier_init(patch, fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        if patch == 0:
            return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
        else:
            return tf.random_uniform((patch, patch, fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
  
    # Variables，训练参数
    # 第一卷积层
    conv1_1_kernel = tf.Variable(xavier_init(patch_size, num_channels, depth_1))
    conv1_1_biases = tf.Variable(tf.zeros([depth_1]))
    conv1_2_kernel = tf.Variable(xavier_init(patch_size, depth_1, depth_1))
    conv1_2_biases = tf.Variable(tf.zeros([depth_1]))
    # 第二卷积层
    conv2_1_kernel = tf.Variable(xavier_init(patch_size, depth_1, depth_2))
    conv2_1_biases = tf.Variable(tf.zeros([depth_2]))
    conv2_2_kernel = tf.Variable(xavier_init(patch_size, depth_2, depth_2))
    conv2_2_biases = tf.Variable(tf.zeros([depth_2]))
    # 第三全连接层
    fc3_weights = tf.Variable(xavier_init(0, (image_size // 4 + 1) * (image_size // 4 + 1) * depth_2, num_hidden))
    fc3_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden]))
    # 第四全连接层
    fc4_weights = tf.Variable(xavier_init(0, num_hidden, num_hidden))
    fc4_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden]))
    # 第五全连接层
    fc5_weights = tf.Variable(xavier_init(0, num_hidden, num_labels))
    fc5_biases = tf.Variable(tf.constant(0.0, shape=[num_labels]))
    
    # 打印每一层的输出
    def print_activations(t):
        print(t.op.name, ' ', t.get_shape().as_list())
      
    # Model.
    def model(data):
        # 第一层卷积层，包含池化层
        with tf.name_scope('conv1') as scope:
            conv = tf.nn.conv2d(data, conv1_1_kernel, [1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + conv1_1_biases)
            conv = tf.nn.conv2d(activation, conv1_2_kernel, [1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + conv1_2_biases)
            conv1 = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)
            print_activations(conv1)
        # 第二层卷积层，包含池化层
        with tf.name_scope('conv2') as scope:
            conv = tf.nn.conv2d(conv1, conv2_1_kernel, [1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + conv2_1_biases)
            conv = tf.nn.conv2d(activation, conv2_2_kernel, [1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + conv2_2_biases)
            conv2 = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)
            print_activations(conv2)
        # 第三层全连接层
        with tf.name_scope('fc3') as scope:
            shape = conv2.get_shape().as_list()
            reshape = tf.reshape(conv2, [-1, shape[1] * shape[2] * shape[3]])
            drop3 = tf.nn.relu(tf.matmul(reshape, fc3_weights) + fc3_biases)
            fc3 = tf.nn.dropout(drop3, 0.5, name=scope)
            print_activations(fc3)
        # 第四层全连接层
        with tf.name_scope('fc4') as scope:
            drop4 = tf.nn.relu(tf.matmul(fc3, fc4_weights) + fc4_biases)
            fc4 = tf.nn.dropout(drop4, 0.5, name=scope)
            print_activations(fc4)
        # 第五层全连接层
        with tf.name_scope('fc5') as scope:
            fc5 = tf.matmul(fc4, fc5_weights) + fc5_biases
        return fc5

    def model_test(data):
        # 第一层卷积层，包含池化层
        with tf.name_scope('conv1') as scope:
            conv = tf.nn.conv2d(data, conv1_1_kernel, [1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + conv1_1_biases)
            conv = tf.nn.conv2d(activation, conv1_2_kernel, [1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + conv1_2_biases)
            conv1 = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)
        # 第二层卷积层，包含池化层
        with tf.name_scope('conv2') as scope:
            conv = tf.nn.conv2d(conv1, conv2_1_kernel, [1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + conv2_1_biases)
            conv = tf.nn.conv2d(activation, conv2_2_kernel, [1, 1, 1, 1], padding='SAME')
            activation = tf.nn.relu(conv + conv2_2_biases)
            conv2 = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)
        # 第三层全连接层
        with tf.name_scope('fc3') as scope:
            shape = conv2.get_shape().as_list()
            reshape = tf.reshape(conv2, [-1, shape[1] * shape[2] * shape[3]])
            drop3 = tf.nn.relu(tf.matmul(reshape, fc3_weights) + fc3_biases)
            fc3 = tf.nn.dropout(drop3, 1, name=scope)
        # 第四层全连接层
        with tf.name_scope('fc4') as scope:
            drop4 = tf.nn.relu(tf.matmul(fc3, fc4_weights) + fc4_biases)
            fc4 = tf.nn.dropout(drop4, 1, name=scope)
        # 第五层全连接层
        with tf.name_scope('fc5') as scope:
            fc5 = tf.matmul(fc4, fc5_weights) + fc5_biases
        return fc5

    # Training computation.
    logits = model(tf_train_dataset)
    beta = 0.0005
    l2_loss =  tf.nn.l2_loss(tf.concat(
          [tf.reshape(conv1_1_kernel, [-1]), tf.reshape(conv1_2_kernel, [-1]), 
           tf.reshape(conv2_1_kernel, [-1]), tf.reshape(conv2_2_kernel, [-1]),
           tf.reshape(fc3_weights, [-1]), tf.reshape(fc4_weights, [-1]), tf.reshape(fc5_weights, [-1])], 0))
    loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits), 1)) + beta * l2_loss
    
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.sigmoid(logits)
    valid_prediction = tf.nn.sigmoid(model_test(tf_valid_dataset))
    test_prediction = tf.nn.sigmoid(model_test(tf_test_dataset))
  
    #  predict = tf.nn.softmax(model_test(data))
    predict = tf.nn.sigmoid(model_test(data))
  
    ## 保存模型，生成saver
    tf.add_to_collection('predict', predict)
    saver = tf.train.Saver(max_to_keep=1)

# In[2]:training and test  
num_steps = 10001
train_accuracy = [0.0]
valid_accuracy = [0.0]
valid_batch_size = valid_label.shape[0] // 1000

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    start_time = time.time()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_label.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :]
        batch_labels = train_label[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            accuracy1 = accuracy(predictions, batch_labels)
            accuracy_temp = list()
            for num in range(1000):
                valid_batch_data = valid_dataset[num*valid_batch_size : (num+1)*valid_batch_size, :, :]
                valid_batch_label = valid_label[num*valid_batch_size : (num+1)*valid_batch_size, :]
                predict_temp = session.run(valid_prediction, feed_dict = {tf_valid_dataset: valid_batch_data})
                accuracy_temp.append(accuracy(predict_temp, valid_batch_label))
            accuracy2 = np.nanmean(np.array(accuracy_temp))
            train_accuracy.append(accuracy1)
            valid_accuracy.append(accuracy2)
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch sq_loss: %.4f%%' % accuracy1)
            print('Validation sq_loss: %.4f%%' % accuracy2)
    saver.save(session, os.path.join(path, "CNN_cracks"))
    train_plot = tf.constant(train_accuracy).eval()
    valid_plot = tf.constant(valid_accuracy).eval()
    
    # plot
    plotx = np.arange(0, num_steps, 500)
    plt.plot(plotx, train_plot[1:], color="#348ABD", label="training")
    plt.plot(plotx, valid_plot[1:], color="#A60628", label="validation")
    plt.legend()
    plt.xlabel("Training iterations")
    plt.ylabel("Error rate")
    plt.title("Learning curve")
    print('training time: %d' % (time.time() - start_time))
    
    
    # 单个样本验证
    saver.restore(session, os.path.join(path, "CNN_cracks"))
    x = 0
    y = 0
    block_size = image_size
    pixel_depth = 255.0
    pic = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2GRAY)
    width = pic.shape[0]
    height = pic.shape[1]
    img = np.zeros([width, height])
    img_pad = np.pad(img, half_struct, 'symmetric')
    pic = (pic - pixel_depth / 2) / pixel_depth * 2
    pic_pad = np.pad(pic, half_block, 'symmetric')
    for x in range(width):
        block_tensor = list()
        for y in range(height):
            block = pic_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1]
            block_tensor.append(block)
        block_tensor = np.array(block_tensor)
        block_tensor = np.reshape(block_tensor, [-1, block_size, block_size, num_channels])
        prediction = session.run(predict, feed_dict={data: block_tensor})
        for y in range(height):
            img_pad[x:x + 2*half_struct + 1, y:y + 2*half_struct + 1] += np.reshape(prediction[y], [struct, struct])
    img = img_pad[half_struct:width+half_struct, half_struct:height+half_struct]
    output = img / np.nanmax(img) * 255
    result = Image.fromarray((output).astype(np.uint8)).save('result.png')
#    display(result)
        
session.close()
del session
