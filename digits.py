#!/usr/bin/python3.6
# -*- coding: utf-8 -*-  

"""
Created on Thu May 30 14:20:50 2019
When run the main function repeatedly, it is wise to add 'tf.reset_default_graph()'
at the command windows, as it would clear all the data preserved in the last running.
Otherwise, errors may take place.
@author: lcy
"""
 
import sys
import os
import time
import random
 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
 
from PIL import Image

 
SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 34
iterations = 30
_it = []
_iterate_accuracy = []

SAVER_DIR = "train-saver/digits/"
 
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z")
license_num = ""
time_begin = time.time()
 
 
# define input node
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
 
x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
 
 
# define convolutional function
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')
 
# define connected layer function
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
 
if __name__ =='__main__' and sys.argv[1]=='train':
    # the aim of this loop is to obtain the mounts of photos in the training directories
    input_count = 0
    for i in range(0,NUM_CLASSES):
        dir = './train_images/training-set/%s/' % i           
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1
 
    # define dimensions and length of arrays
    input_images = np.array([[0]*SIZE for i in range(input_count)])
    input_labels = np.array([[0]*NUM_CLASSES for i in range(input_count)])
 
    # the second loop is to generate data of photos and lables
    index = 0
    for i in range(0,NUM_CLASSES):
        dir = './train_images/training-set/%s/' % i         
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # binarization processing the photos in order to make the lines thinner and increase accuracy
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w+h*width] = 0
                        else:
                            input_images[index][w+h*width] = 1
                input_labels[index][i] = 1
                index += 1
 
    # the aim of this loop is to obtain the mounts of photos in the validation directories
    val_count = 0
    for i in range(0,NUM_CLASSES):
        dir = './train_images/validation-set/%s/' % i          
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1
 
    # define dimensions and length of arrays
    val_images = np.array([[0]*SIZE for i in range(val_count)])
    val_labels = np.array([[0]*NUM_CLASSES for i in range(val_count)])
 
    # the second loop is to generate data of photos and lables
    index = 0
    for i in range(0,NUM_CLASSES):
        dir = './train_images/validation-set/%s/' % i          
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) > 230:
                            val_images[index][w+h*width] = 0
                        else:
                            val_images[index][w+h*width] = 1
                val_labels[index][i] = 1
                index += 1
    #remember: DO NOT try to run tensorflow core more than one program in a single computer with only one GPU device.

    # per_process_gpu_memory_fraction: determine the upper limit for each GPU memory.
    # yet it can only function on GPU uniformly.
    # it cannot set individual GPU memory limit for each GPU device .

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config.gpu_options.allow_growth = True
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_soft_placement = True, log_device_placement = True)
    with tf.Session(config=config) as sess: 
        # the first convolutional layer
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv1")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')
 
        # the second convolutional layer
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="W_conv2")
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv2")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')
 
        # connected layer
        W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20*32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)
 
        # dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
        # readout layer
        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name="W_fc2")
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")
 
        # Define optimizers and train Op
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
 
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
        sess.run(tf.global_variables_initializer())
        # print(sess.run(tf.global_variables_initializer()))
        time_elapsed = time.time() - time_begin
        print("reading photos costs: %dsec" % time_elapsed)
        time_begin = time.time()
 
        print ("program has read %s images for training, with %s labels" % (input_count, input_count))
 
        # set the number of op and iterations for each training. in order to enable any given number of total photos, 
        # a remainder is defined here. For instance, when the total number of photos is 150 and each training 60 photos, 
        # then the remainder is 30.

        batch_size = 60
        iterations = iterations
        batches_count = int(input_count / batch_size)
        remainder = input_count % batch_size
        print ("training dataset is divided into %s groups, with former groups combined with %s data points and last groups %s data points" % (batches_count+1, batch_size, remainder))
 
        # execute iterations-training loop
        for it in range(iterations):
            # 这里的关键是要把输入数组转为np.array
            for n in range(batches_count):
                train_step.run(feed_dict={x: input_images[n*batch_size:(n+1)*batch_size], y_: input_labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})
            if remainder > 0:
                start_index = batches_count * batch_size;
                train_step.run(feed_dict={x: input_images[start_index:input_count-1], y_: input_labels[start_index:input_count-1], keep_prob: 0.5})
 
            # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
            iterate_accuracy = 0
            if it%5 == 0:
                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                _it.append(it)
                _iterate_accuracy.append(iterate_accuracy)
                print ('the %d times iterating accuracy: %0.5f%%' % (it, iterate_accuracy*100))
                if iterate_accuracy >= 0.9999 and it >= iterations:
                    break;
 
        print ('training finished!')
        time_elapsed = time.time() - time_begin
        print ("training costs %d sec" % time_elapsed)
        time_begin = time.time()

        plt.plot(_it, _iterate_accuracy, label = 'accuracy %')
        plt.xlabel('iterations')
        plt.ylabel('accuracy %')
        plt.title("digits-train-accuracy")
        plt.savefig('./digits-train-accuracy')
        plt.show()
 
        # save training results
        if not os.path.exists(SAVER_DIR):
            print ('There do not exsit directories to save training results. Now create the directories.')
            os.makedirs(SAVER_DIR)
        # initialize saver
        saver = tf.train.Saver()            
        saver_path = saver.save(sess, "%smodel.ckpt"%(SAVER_DIR))
 
 
 
if __name__ =='__main__' and sys.argv[1]=='predict':
    # open the training data
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta"%(SAVER_DIR))

    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config.gpu_options.allow_growth = True
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_soft_placement = True, log_device_placement = True)
    with tf.Session(config=config) as sess: 
        model_file=tf.train.latest_checkpoint(SAVER_DIR)
        saver.restore(sess, model_file)
 
        # the first convolutional layer
        W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
        b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')
 
        # the second convolutional layer
        W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
        b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')
 
 
        # connected layer
        W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
        b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20*32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)
 
        # dropout
        keep_prob = tf.placeholder(tf.float32)
 
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
        # readout layer
        W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
        b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")
 
        # Define optimizers and train Op
        conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
        for n in range(3,8):
            path = "test_images/%s.bmp" % (n)
            img = Image.open(path)
            width = img.size[0]
            height = img.size[1]
 
            img_data = [[0]*SIZE for i in range(1)]
            for h in range(0, height):
                for w in range(0, width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w+h*width] = 1
                    else:
                        img_data[0][w+h*width] = 0
            
            result = sess.run(conv, feed_dict = {x: np.array(img_data), keep_prob: 1.0})
            
            max1 = 0
            max2 = 0
            max3 = 0
            max1_index = 0
            max2_index = 0
            max3_index = 0
            for j in range(NUM_CLASSES):
                if result[0][j] > max1:
                    max1 = result[0][j]
                    max1_index = j
                    continue
                if (result[0][j]>max2) and (result[0][j]<=max1):
                    max2 = result[0][j]
                    max2_index = j
                    continue
                if (result[0][j]>max3) and (result[0][j]<=max2):
                    max3 = result[0][j]
                    max3_index = j
                    continue
            
            license_num = license_num + LETTERS_DIGITS[max1_index]
            print ("possibility  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (LETTERS_DIGITS[max1_index],max1*100, LETTERS_DIGITS[max2_index],max2*100, LETTERS_DIGITS[max3_index],max3*100))
            
        print ("car number: [%s]" % license_num)
