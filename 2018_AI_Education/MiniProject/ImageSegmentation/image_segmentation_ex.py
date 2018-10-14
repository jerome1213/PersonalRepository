from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import functools

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib as mpl

mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)

from sklearn.model_selection import train_test_split
from PIL import Image
from IPython.display import clear_output

import tensorflow as tf

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_dir = 'sd_train'
img_dir = os.path.join(dataset_dir, "train")
label_dir = os.path.join(dataset_dir, "train_labels")

x_train_filenames = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
x_train_filenames.sort()

y_train_filenames = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir)]
y_train_filenames.sort()

x_train_filenames, x_test_filenames, y_train_filenames, y_test_filenames = \
    train_test_split(x_train_filenames, y_train_filenames, test_size=0.2, random_state=219)

num_train_examples = len(x_train_filenames)
num_test_examples = len(x_test_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))

x_train_filenames[:10]
y_train_filenames[:10]
y_test_filenames[:10]

display_num = 5

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

# Set hyperparameters
image_size = 64
img_shape = (image_size, image_size, 3)
batch_size = 8
max_epochs = 100
print_steps = 50
save_epochs = 20
train_dir = 'train/exp1'


def _process_pathnames(fname, label_path):
    # We map this function onto each pathname pair
    img_str = tf.read_file(fname)
    img = tf.image.decode_bmp(img_str, channels=3)
    
    label_img_str = tf.read_file(label_path)
    label_img = tf.image.decode_bmp(label_img_str, channels=1)
    
    resize = [image_size, image_size]
    img = tf.image.resize_images(img, resize)
    label_img = tf.image.resize_images(label_img, resize)
    
    scale = 1 / 255.
    img = tf.to_float(img) * scale
    label_img = tf.to_float(label_img) * scale
    
    return img, label_img


def get_baseline_dataset(filenames,
                         labels,
                         threads=5,
                         batch_size=batch_size,
                         max_epochs=max_epochs,
                         shuffle=True):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    # [CHECK] Data Augmentation 작업 필요함!!
    
    if shuffle:
        dataset = dataset.shuffle(num_x * 10)
    
    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat(max_epochs).batch(batch_size)
    return dataset


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
    # plt.show()

from ImageSegmentation.model_config import ModelConfig
from ImageSegmentation.train_config import TrainConfig

model_config = ModelConfig()
train_config = TrainConfig()


## Build the model
def conv_block(inputs, num_outputs, is_training, scope):
    batch_norm_params = {'decay': 0.9,
                         'epsilon': 0.001,
                         'is_training': is_training,
                         'scope': 'batch_norm'}
    with tf.variable_scope(name_or_scope=scope, values=[inputs]):
        with slim.arg_scope([slim.conv2d],
                            num_outputs=num_outputs,
                            kernel_size=model_config.con_kernel_shape['conv_kernel_size'],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            # encoder = slim.conv2d(inputs, scope='conv1')
            # encoder = slim.conv2d(encoder, scope='conv2')
            encoder = inputs
            encoder = slim.repeat(encoder, 2, slim.conv2d, scope='conv')
            return encoder


def encoder_block(inputs, num_outputs, is_training, scope):
    with tf.variable_scope(name_or_scope=scope, values=[inputs]):
        encoder = conv_block(inputs, num_outputs, is_training, scope)
        encoder_pool = slim.max_pool2d(encoder, kernel_size=model_config.encoder_kernel_shape['pool_kernel_size'], scope='pool')
    return encoder_pool, encoder


def decoder_block(inputs, concat_tensor, num_outputs, is_training, scope):
    '''
    C -> conv2d_trans -. C/2 => tf.concat -> C -> conv2d -> C
    '''
    batch_norm_params = {'decay': 0.9,
                         'epsilon': 0.001,
                         'is_training': is_training,
                         'scope': 'batch_norm'}
    with tf.variable_scope(name_or_scope=scope, values=[inputs]):
        decoder = slim.conv2d_transpose(inputs=inputs,
                                        num_outputs=np.floor(num_outputs / 2),
                                        stride=model_config.decoder_kernel_shape['convT_kernel_stride'],
                                        kernel_size=model_config.decoder_kernel_shape['convT_kernel_size'],
                                        activation_fn=None, scope='convT')
        decoder = tf.concat([decoder, concat_tensor], axis=3)
        decoder = slim.batch_norm(decoder, **batch_norm_params)
        decoder = tf.nn.relu(decoder)
        decoder = conv_block(decoder, num_outputs=num_outputs, is_training=is_training, scope='dec' + scope)
    return decoder


class UNet(object):
    def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds
    
    def build_images(self):
        # tf.data.Iterator.from_string_handle의 output_shapes는 default = None이지만 꼭 값을 넣는 게 좋음
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle,
                                                            self.train_ds.output_types,
                                                            self.train_ds.output_shapes)
        self.input_images, self.targets = self.iterator.get_next()
    
    def inference(self, inputs, is_training, reuse=False):
        with tf.variable_scope('', reuse=reuse) as scope:
            # inputs: [64, 64, 3]
            encoder0_pool, encoder0 = encoder_block(inputs=inputs, num_outputs=model_config.chout['enc0'], is_training=is_training, scope='enc0')
            # encoder0_pool: [32, 32, 32]
            encoder1_pool, encoder1 = encoder_block(inputs=encoder0_pool, num_outputs=model_config.chout['enc1'], is_training=is_training, scope='enc1')
            # encoder1_pool: [16, 16, 64]
            encoder2_pool, encoder2 = encoder_block(inputs=encoder1_pool, num_outputs=model_config.chout['enc2'], is_training=is_training, scope='enc2')
            # encoder2_pool: [8, 8, 128]
            encoder3_pool, encoder3 = encoder_block(inputs=encoder2_pool, num_outputs=model_config.chout['enc3'], is_training=is_training, scope='enc3')
            # encoder3_pool: [4, 4, 256]
            
            center = conv_block(encoder3_pool, num_outputs=model_config.chout['center'], is_training=is_training, scope='center')
            # center: [1, 1, 512]
            
            decoder3 = decoder_block(center, encoder3, num_outputs=model_config.chout['dec3'], is_training=is_training, scope='dec3')
            # decoder3 = [4, 4, 256]
            decoder2 = decoder_block(decoder3, encoder2, num_outputs=model_config.chout['dec2'], is_training=is_training, scope='dec2')
            # decoder2 = [8, 8, 128]
            decoder1 = decoder_block(decoder2, encoder1, num_outputs=model_config.chout['dec1'], is_training=is_training, scope='dec1')
            # decoder1 = [16, 16, 64]
            decoder0 = decoder_block(decoder1, encoder0, num_outputs=model_config.chout['dec0'], is_training=is_training, scope='dec0')
            # decoder0 = [32, 32, 32]
            
            logits = slim.conv2d(decoder0, num_outputs=1, kernel_size=[1, 1], activation_fn=None, scope='outputs')
            
            return logits
    
    def dice_coeff(self, y_true, y_logits):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(tf.nn.sigmoid(y_logits), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    def dice_loss(self, y_true, y_logits):
        loss = 1 - self.dice_coeff(y_true, y_logits)
        return loss
    
    def bce_dice_loss(self, y_true, y_logits):
        loss = tf.losses.sigmoid_cross_entropy(y_true, y_logits) + self.dice_loss(y_true, y_logits)
        # sigmoid_cross_entropy :: 0 or 1 을 구분 하기 위해
        return loss
    
    def build(self):
        self.global_step = slim.get_or_create_global_step()
        
        self.build_images()
        self.logits = self.inference(self.input_images, is_training=True)
        self.logits_val = self.inference(self.input_images, is_training=False, reuse=True)
        self.predicted_images = tf.nn.sigmoid(self.logits_val)
        
        self.loss = self.bce_dice_loss(self.targets, self.logits)
        
        print("complete model build.")


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
        
        epochs = global_step_ * batch_size / float(num_train_examples)
        duration = time.time() - start_time
        
        if global_step_ % print_steps == 0:
            clear_output(wait=True)
            examples_per_sec = batch_size / float(duration)
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
            # plt.show()
        
        # save model checkpoint periodically
        if int(epochs) % save_epochs == 0 and pre_epochs != int(epochs):
            tf.logging.info('Saving model with global step {} (= {} epochs) to disk.'.format(global_step_, int(epochs)))
            saver.save(sess, train_dir + 'model.ckpt', global_step=global_step_)
            pre_epochs = int(epochs)
    
    except tf.errors.OutOfRangeError:
        print("End of dataset")  # ==> "End of dataset"
        tf.logging.info('Saving model with global step {} (= {} epochs) to disk.'.format(global_step_, int(epochs)))
        saver.save(sess, train_dir + 'model.ckpt', global_step=global_step_)
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
