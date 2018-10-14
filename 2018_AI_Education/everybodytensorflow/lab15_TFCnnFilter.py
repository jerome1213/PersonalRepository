# -*- coding: utf-8 -*-
# ! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: lan15_runCNNFilter.py

    Generation of exemplary feature of convolutional neural network
    using two types of filters: horizontal and vertical filters

    This example is originally given by
    A. Geron "Hands-On Machine Learning with Scikit-Learn and TensorFlow",
    O'REILLY 2017. page363

    written by Jaewook Kang @ Nov 2017
#------------------------------------------------------------
'''
import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
import matplotlib.pyplot as plt

# load images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)

# load images data set size
batch_size, height, width, channels = dataset.shape

# create two 7 X 7 filters
filter_size = 3
num_of_filters = 2
filters = np.zeros(shape=(filter_size, filter_size, channels, num_of_filters), dtype=np.float32)

filters[:, 1, :, 0] = 1  # vertical line filters
filters[1, :, :, 1] = 1  # horizontal line filters

# Creat a graph with with input X plus a convolutional layer
# applying the 2 filter defined abovez
# N>H>W>C
X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='input')

stride = 2
# prediction CNN with two filters and input X
# X is the input mini-batch
# 7 X 7 filters is the set of filters to apply
# padding = 'SAME', which means the conv layer use zero padding to keep the same feature map size
# padding = 'VALID', which mean the conv layer does not use zero padding
# conv_output = tf.nn.conv2d(X, filters, strides=[1, stride, stride, 1], padding='SAME')
conv_output = tf.nn.conv2d(X, filters, strides=[1, stride, stride, 1], padding='VALID')

with tf.Session() as sess:
    output = sess.run(conv_output, feed_dict={X: dataset})

hfig = plt.figure(1, figsize=(5, 10))

plt.subplot(4, 2, 1)
plt.imshow(china)
plt.title('The original china')

plt.subplot(4, 2, 2)
plt.imshow(flower)
plt.title('The original flower')


plt.subplot(4, 2, 3)
plt.imshow(filters[:, :, :, 0])
plt.title('Vertical filter')

plt.subplot(4, 2, 4)
plt.imshow(filters[:, :, :, 1])
plt.title('Horizontal filter')

# plot 1st image's and feature map with vertical filter
plt.subplot(4, 2, 5)
plt.imshow(output[0, :, :, 0], cmap='gray')

# plot 1st image's and feature map with horizontal filter
plt.subplot(4, 2, 6)
plt.imshow(output[0, :, :, 1], cmap='gray')

# plot 2nd image's and feature map with vertical filter
plt.subplot(4, 2, 7)
plt.imshow(output[1, :, :, 0], cmap='gray')

# plot 2nd image's and feature map with horizontal filter
plt.subplot(4, 2, 8)
plt.imshow(output[1, :, :, 1], cmap='gray')

plt.show()
