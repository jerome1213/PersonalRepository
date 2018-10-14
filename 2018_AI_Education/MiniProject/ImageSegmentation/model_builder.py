from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ImageSegmentation.model_config import ModelConfig
from ImageSegmentation.train_config import TrainConfig

model_config = ModelConfig()
train_config = TrainConfig()


## Build the model
def conv_block(inputs, num_outputs, is_training, scope):
    with tf.variable_scope(name_or_scope=scope, values=[inputs]):
        with slim.arg_scope([slim.conv2d],
                            num_outputs=num_outputs,
                            kernel_size=model_config.con_kernel_shape['conv_kernel_size'],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=get_batch_norm_params(is_training)):
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
    with tf.variable_scope(name_or_scope=scope, values=[inputs]):
        decoder = slim.conv2d_transpose(inputs=inputs,
                                        num_outputs=np.floor(num_outputs / 2),
                                        stride=model_config.decoder_kernel_shape['convT_kernel_stride'],
                                        kernel_size=model_config.decoder_kernel_shape['convT_kernel_size'],
                                        activation_fn=None, scope='convT')
        decoder = tf.concat([decoder, concat_tensor], axis=3)
        decoder = slim.batch_norm(decoder, **get_batch_norm_params(is_training))
        decoder = tf.nn.relu(decoder)
        decoder = conv_block(decoder, num_outputs=num_outputs, is_training=is_training, scope='dec' + scope)
    return decoder


def get_batch_norm_params(is_training):
    batch_norm_params = {'decay': 0.9,
                         'epsilon': 0.001,
                         'is_training': is_training,
                         'scope': 'batch_norm'}
    return batch_norm_params
