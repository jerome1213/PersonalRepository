import tensorflow as tf


class ModelConfig(object):
    def __init__(self):
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = None,
        
        self.chout = { \
            'enc0': 32,
            'enc1': 64,
            'enc2': 128,
            'enc3': 256,
            'center': 512,
            'dec3': 256,
            'dec2': 128,
            'dec1': 62,
            'dec0': 32,
            'logit': 1
        }
        
        self.con_kernel_shape = { \
            'conv_kernel_size': [3, 3],
            'conv_kernel_stride': [1, 1]
        }
        
        self.decoder_kernel_shape = { \
            'convT_kernel_size': [2, 2],
            'convT_kernel_stride': [2, 2],
            'conv_kernel_size': [3, 3],
            'conv_kernel_stride': [1, 1]
        }
        
        self.encoder_kernel_shape = { \
            'conv_kernel_size': [3, 3],
            'conv_kernel_stride': [1, 1],
            'pool_kernel_size': [2, 2],
            'pool_kernel_stride': [2, 2]
        }
