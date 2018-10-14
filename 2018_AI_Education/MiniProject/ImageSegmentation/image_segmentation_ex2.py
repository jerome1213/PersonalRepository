from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import ImageSegmentation.UNet as UNet

mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)

from PIL import Image
from IPython.display import clear_output

import tensorflow as tf
from ImageSegmentation.model_config import ModelConfig
from ImageSegmentation.train_config import TrainConfig

model_config = ModelConfig()
train_config = TrainConfig()
slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

r_choices = np.random.choice(num_train_examples, display_num)

plt.figure(figsize=(10, 15))
for i in range(0, display_num * 2, 2):
    img_num = r_choices[i // 2]
    x_pathname = x_train_filenames[img_num]
    y_pathname = y_train_filenames[img_num]
    
    plt.subplot(display_num, 2, i + 1)
    plt.imshow(Image.open(x_pathname))
    plt.title("Original Image")
    
    example_labels = Image.open(y_pathname)
    label_vals = np.unique(example_labels)
    
    plt.subplot(display_num, 2, i + 2)
    plt.imshow(example_labels)
    plt.title("Masked Image")

plt.suptitle("Examples of Images and their Masks")
# plt.show()


train_ds = get_baseline_dataset(x_train_filenames,
                                y_train_filenames)
test_ds = get_baseline_dataset(x_test_filenames,
                               y_test_filenames,
                               shuffle=False)

print(train_ds)

temp_ds = get_baseline_dataset(x_train_filenames,
                               y_train_filenames,
                               batch_size=1,
                               max_epochs=1,
                               shuffle=False)
# Let's examine some of these augmented images
temp_iter = temp_ds.make_one_shot_iterator()
next_element = temp_iter.get_next()
with tf.Session() as sess:
    batch_of_imgs, label = sess.run(next_element)
    
    # Running next element in our graph will produce a batch of images
    plt.figure(figsize=(10, 10))
    img = batch_of_imgs[0]
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    
    plt.subplot(1, 2, 2)
    plt.imshow(label[0, :, :, 0])
    plt.show()

model = UNet(train_ds=train_ds, test_ds=test_ds)
model.build()

# show info for trainable variables
t_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(t_vars, print_info=True)

opt = tf.train.AdamOptimizer(learning_rate=2e-4)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    opt_op = opt.minimize(model.loss, global_step=model.global_step)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
tf.logging.info('Start Session.')

train_iterator = train_ds.make_one_shot_iterator()
train_handle = sess.run(train_iterator.string_handle())
test_iterator = test_ds.make_one_shot_iterator()
test_handle = sess.run(test_iterator.string_handle())

# save loss values for plot
loss_history = []
pre_epochs = 0
while True:
    try:
        start_time = time.time()
        _, global_step_, loss = sess.run([opt_op,
                                          model.global_step,
                                          model.loss],
                                         feed_dict={model.handle: train_handle})
        
        epochs = global_step_ * train_config.batch_size / float(num_train_examples)
        duration = time.time() - start_time
        
        if global_step_ % train_config.print_steps == 0:
            clear_output(wait=True)
            examples_per_sec = train_config.batch_size / float(duration)
            print("Epochs: {:.2f} global_step: {} loss: {:.3f} ({:.2f} examples/sec; {:.3f} sec/batch)".format(
                epochs, global_step_, loss, examples_per_sec, duration))
            
            loss_history.append([epochs, loss])
            
            # print sample image
            img, label, predicted_label = sess.run([model.input_images, model.targets, model.predicted_images],
                                                   feed_dict={model.handle: test_handle})
            plt.figure(figsize=(10, 20))
            plt.subplot(1, 3, 1)
            plt.imshow(img[0, :, :, :])
            plt.title("Input image")
            
            plt.subplot(1, 3, 2)
            plt.imshow(label[0, :, :, 0])
            plt.title("Actual Mask")
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_label[0, :, :, 0])
            plt.title("Predicted Mask")
            plt.show()
        
        # save model checkpoint periodically
        if int(epochs) % train_config.save_epochs == 0 and pre_epochs != int(epochs):
            tf.logging.info('Saving model with global step {} (= {} epochs) to disk.'.format(global_step_, int(epochs)))
            saver.save(sess, train_config.train_dir + 'model.ckpt', global_step=global_step_)
            pre_epochs = int(epochs)
    
    except tf.errors.OutOfRangeError:
        print("End of dataset")  # ==> "End of dataset"
        tf.logging.info('Saving model with global step {} (= {} epochs) to disk.'.format(global_step_, int(epochs)))
        saver.save(sess, train_config.train_dir + 'model.ckpt', global_step=global_step_)
        break

tf.logging.info('complete training...')

loss_history = np.asarray(loss_history)
plt.plot(loss_history[:, 0], loss_history[:, 1])
plt.show()

test_ds_eval = get_baseline_dataset(x_test_filenames,
                                    y_test_filenames,
                                    batch_size=num_test_examples,
                                    max_epochs=1,
                                    shuffle=False)

test_iterator_eval = test_ds_eval.make_one_shot_iterator()
test_handle_eval = sess.run(test_iterator_eval.string_handle())

mean_iou, mean_iou_op = tf.metrics.mean_iou(labels=tf.to_int32(tf.round(model.targets)),
                                            predictions=tf.to_int32(tf.round(model.predicted_images)),
                                            num_classes=2,
                                            name='mean_iou')
sess.run(tf.local_variables_initializer())

sess.run(mean_iou_op, feed_dict={model.handle: test_handle_eval})
print("mean iou:", sess.run(mean_iou))

test_ds_visual = get_baseline_dataset(x_test_filenames,
                                      y_test_filenames,
                                      batch_size=1,
                                      max_epochs=1,
                                      shuffle=False)

test_iterator_visual = test_ds_visual.make_one_shot_iterator()
test_handle_visual = sess.run(test_iterator_visual.string_handle())

# Let's visualize some of the outputs

# Running next element in our graph will produce a batch of images
plt.figure(figsize=(10, 20))
for i in range(5):
    # img, label, predicted_label = sess.run([model.input_images, model.targets, model.predicted_images],
    img, label, predicted_label = sess.run([model.input_images,
                                            tf.to_int32(tf.round(model.targets)),
                                            tf.to_int32(tf.round(model.predicted_images))],
                                           feed_dict={model.handle: test_handle_visual})
    
    plt.subplot(5, 3, 3 * i + 1)
    plt.imshow(img[0, :, :, :])
    plt.title("Input image")
    
    plt.subplot(5, 3, 3 * i + 2)
    plt.imshow(label[0, :, :, 0])
    plt.title("Actual Mask")
    
    plt.subplot(5, 3, 3 * i + 3)
    plt.imshow(predicted_label[0, :, :, 0])
    plt.title("Predicted Mask")
plt.suptitle("Examples of Input Image, Label, and Prediction")
plt.show()
