from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from ImageSegmentation.train_config import TrainConfig

train_config = TrainConfig()


class ISDataLoader(object):
    
    def __init__(self):
        self.dataset_dir = 'sd_train'
        self.img_dir = os.path.join(self.dataset_dir, "train")
        self.label_dir = os.path.join(self.dataset_dir, "train_labels")
        
        self.x_train_filenames = [os.path.join(self.img_dir, filename) for filename in os.listdir(self.img_dir)]
        self.y_train_filenames = [os.path.join(self.label_dir, filename) for filename in os.listdir(self.label_dir)]
        
        self.x_train_filenames.sort()
        self.y_train_filenames.sort()
        
        self.x_train_filenames, self.x_test_filenames, self.y_train_filenames, self.y_test_filenames = \
            train_test_split(self.x_train_filenames, self.y_train_filenames, test_size=0.2, random_state=219)
        
        self.num_train_examples = len(self.x_train_filenames)
        self.num_test_examples = len(self.x_test_filenames)
        
        self.num_train_examples = len(self.x_train_filenames)
        self.num_test_examples = len(self.x_test_filenames)
        
        self.x_train_filenames[:10]
        self.y_train_filenames[:10]
        self.y_test_filenames[:10]
        
        self.display_num = 5
        print("Number of training examples: {}".format(self.num_train_examples))
        print("Number of test examples: {}".format(self.num_test_examples))
    
    def _process_pathnames(self, fname, label_path):
        # We map this function onto each pathname pair
        
        img_str = tf.read_file(fname)
        img = tf.image.decode_bmp(img_str, channels=3)
        
        label_img_str = tf.read_file(label_path)
        label_img = tf.image.decode_bmp(label_img_str, channels=1)
        
        resize = [train_config.image_size, train_config.image_size]
        img = tf.image.resize_images(img, resize)
        label_img = tf.image.resize_images(label_img, resize)
        
        scale = 1 / 255.
        img = tf.to_float(img) * scale
        label_img = tf.to_float(label_img) * scale
        
        return img, label_img
    
    def get_baseline_dataset(self, filenames,
                             labels,
                             threads=5,
                             batch_size=train_config.batch_size,
                             max_epochs=train_config.max_epochs,
                             shuffle=True):
        num_x = len(filenames)
        # Create a dataset from the filenames and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        # Map our preprocessing function to every element in our dataset, taking
        # advantage of multithreading
        dataset = dataset.map(self._process_pathnames, num_parallel_calls=threads)
        # [CHECK] Data Augmentation 작업 필요함!!
        
        if shuffle:
            dataset = dataset.shuffle(num_x * 10)
        
        # It's necessary to repeat our data for all epochs
        dataset = dataset.repeat(max_epochs).batch(batch_size)
        return dataset
