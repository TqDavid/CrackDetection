# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:16:12 2017

由于代码放在服务器，python版本和tensorflow版本不一样，此代码为服务器兼容的版本

python: 2.7
tensoflow: 0.12.1

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
from scipy import ndimage

# In[0]: load images as train, validation and test dataset

# 数据库路径
#path = 'G:/Crack Detection/project/database/pydatabase'
#sample = "G:/Crack Detection/project/database/pydatabase/image/001.jpg"
path = '../pydatabase'
sample = "../pydatabase/image/001.jpg"

# variable
num_channels = 3
image_size = 27
image_width = 320  # Pixel width and height.
image_height = 480
half_block = image_size // 2
pixel_depth = 255.0
num_labels = 2

batch_size = 256
patch_size = 3
depth_1 = 64
depth_2 = 128
num_hidden = 512

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
    return dataset

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
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels    

# 分离样本
def depart(image_folder):
    GT_folder = image_folder + '_GT'
    datapos = list()
    dataneg = list()
    images = load_image(image_folder)
    labels = load_GT(GT_folder)
    pos_block = 0
    neg_block = 0
    for num_image in range(np.shape(images)[0]):
        image = images[num_image, :, :, :]
        image_pad = np.zeros([image_width+image_size-1, image_height+image_size-1, 3])
        for i in range(3):
            image_pad[:, :, i] = np.pad(image[:, :, i], half_block, 'constant', constant_values=0)
        for x in range(0, image_width, 3):
            for y in range(0, image_height, 3):
                if x - half_block < 0 or y - half_block < 0 or x + half_block >= image_width or y + half_block >= image_height:
                    block = image_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1, :]
                else:
                    block = image[x-half_block:x+half_block+1, y-half_block:y+half_block+1, :]    
                if labels[num_image, x, y] != 0:
#                    b_show = Image.fromarray(block.astype(np.uint8))
#                    display(b_show)
#                    l_show = labels[num_image, x-half_block:x+half_block, y-half_block:y+half_block]
#                    l_show = Image.fromarray(l_show.astype(np.uint8))
#                    display(l_show)
                    datapos.append(block)
                    pos_block += 1
                    
        for x in range(0, image_width, 3):
            for y in range(0, image_height, 3):
                if x - half_block < 0 or y - half_block < 0 or x + half_block >= image_width or y + half_block >= image_height:
                    block = image_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1, :]
                else:
                    block = image[x-half_block:x+half_block+1, y-half_block:y+half_block+1, :]    
                if labels[num_image, x, y] == 0 and neg_block < 2*pos_block:
                    dataneg.append(block)
                    neg_block += 1
    label_pos = list(np.ones(len(datapos)))
    label_neg = list(np.zeros(len(dataneg)))
    label = label_pos + label_neg
    label = [int(i) for i in label]
    dataset = datapos + dataneg
    del datapos, dataneg
    dataset_ = np.array(dataset)
    label_ = np.array(label)
    dataset__ = (dataset_ - pixel_depth) / pixel_depth
    dataset___, label__ = randomize(dataset__, label_)
    dataset____, label___ = reformat(dataset___, label__)
    print("data positive: %d" % pos_block)
    print("data negative: %d" % neg_block)
    return dataset____, label___

test_dataset, test_label = depart(os.path.join(path, 'test'))
valid_dataset, valid_label = depart(os.path.join(path, 'validation'))
train_dataset, train_label = depart(os.path.join(path, 'train'))

print('Training set', train_dataset.shape, train_label.shape)
print('Validation set', valid_dataset.shape, valid_label.shape)
print('Test set', test_dataset.shape, test_label.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# In[2]: model   

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
    tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)
    
    # 定义用于测试用的变量，并加入存储
    data = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels))
    tf.add_to_collection('data', data)
  
    # Variables，训练参数
    # 第一卷积层
    conv1_1_kernel = tf.Variable(tf.truncated_normal([4, 4, num_channels, depth_1], stddev=0.1))
    conv1_1_biases = tf.Variable(tf.zeros([depth_1]))
    conv1_2_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_1, depth_1], stddev=0.1))
    conv1_2_biases = tf.Variable(tf.zeros([depth_1]))
    # 第二卷积层
    conv2_1_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_1, depth_2], stddev=0.1))
    conv2_1_biases = tf.Variable(tf.zeros([depth_2]))
    conv2_2_kernel = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_2, depth_2], stddev=0.1))
    conv2_2_biases = tf.Variable(tf.zeros([depth_2]))
    # 第三全连接层
    fc3_weights = tf.Variable(tf.truncated_normal(
          [(image_size // 4 + 1) * (image_size // 4 + 1) * depth_2, num_hidden], stddev=0.1))
    fc3_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden]))
    # 第四全连接层
    fc4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
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
            reshape = tf.reshape(conv2, [shape[0], shape[1] * shape[2] * shape[3]])
            drop3 = tf.nn.relu(tf.matmul(reshape, fc3_weights) + fc3_biases)
            fc3 = tf.nn.dropout(drop3, 1, name=scope)
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
            reshape = tf.reshape(conv2, [shape[0], shape[1] * shape[2] * shape[3]])
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
    beta = 0#0.0005
    l2_loss =  tf.nn.l2_loss(tf.concat(0,
          [tf.reshape(conv1_1_kernel, [-1]), tf.reshape(conv1_2_kernel, [-1]), 
           tf.reshape(conv2_1_kernel, [-1]), tf.reshape(conv2_2_kernel, [-1]),
           tf.reshape(fc3_weights, [-1]), tf.reshape(fc4_weights, [-1]), tf.reshape(fc5_weights, [-1])]))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + beta * l2_loss
    
    # Optimizer.
    optimizer = tf.train.AdadeltaOptimizer(1).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model_test(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model_test(tf_test_dataset))
  
    #  predict = tf.nn.softmax(model_test(data))
    predict = tf.argmax(tf.nn.softmax(model_test(data)), 0)
  
    ## 保存模型，生成saver
    tf.add_to_collection('predict', predict)
    saver = tf.train.Saver(max_to_keep=1)

# In[2]:training and test  
num_steps = 20001
train_accuracy = [0.0]
valid_accuracy = [0.0]

width = 320
height = 480

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    start_time = time.time()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_label.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_label[offset:(offset + batch_size)]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            accuracy1 = accuracy(predictions, batch_labels)
            accuracy2 = accuracy(valid_prediction.eval(), valid_label)
            train_accuracy.append(accuracy(predictions, batch_labels))
            valid_accuracy.append(accuracy(valid_prediction.eval(), valid_label))
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_label))
    saver.save(session, "../pydatabase/CNN_cracks")
    train_plot = tf.constant(train_accuracy).eval()
    valid_plot = tf.constant(valid_accuracy).eval()
    
    
    # 单个样本验证
    x = 0
    y = 0
    block_size = image_size
    slide = 15
    pixel_depth = 255.0
    pic = cv2.imread(sample)
    img = np.zeros(width, height)
    pic = (pic - pixel_depth / 2) / pixel_depth
    pic_pad = np.zeros([width+image_size-1, height+image_size-1, 3])
    for i in range(3):
        pic_pad[:, :, i] = np.pad(pic, half_block, 'constant', constant_values=0)
    for x in range(width):
        for y in range(height):
            if x - half_block < 0 or y - half_block < 0 or x + half_block >= width or y + half_block >= height:
                block = pic_pad[x:x + 2*half_block + 1, y:y + 2*half_block + 1, :]
            else:
                block = pic[x-half_block:x+half_block+1, y-half_block:y+half_block+1, :]
            prediction = session.run(predict, feed_dict={data: block})
            if prediction == 1:
                img[x,y] = 1
    result = Image.fromarray(img.astype(np.uint8))
    display(result)
    
    plotx = np.arange(0, num_steps, 100)
    plt.plot(plotx, 100 - train_plot[1:], color="#348ABD", label="training")
    plt.plot(plotx, 100 - valid_plot[1:], color="#A60628", label="validation")
    plt.legend()
    plt.xlabel("Training steps")
    plt.ylabel("Error rate")
    plt.title("Learning curve")
#    saver.restore(session, "../pydatabase/CNN_cracks")
    print(time.time() - start_time)
#    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    
session.close()
del session