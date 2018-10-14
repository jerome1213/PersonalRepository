import tensorflow as tf
import tensorflow.contrib.slim as slim
from ImageSegmentation.model_builder import conv_block, encoder_block, decoder_block

from ImageSegmentation.model_config import ModelConfig

model_config = ModelConfig()


class UNet(object):
    def __init__(self, train_ds, test_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.depth = 4
    
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
    
    def inference2(self, inputs, is_training, reuse=False):
        encoders = []
        with tf.variable_scope('', reuse=reuse) as scope:
            for step in range(self.depth):
                input = inputs if step == 0 else encoders[step - 1][0]
                encoder_pool, encoder = encoder_block(inputs=input, num_outputs=model_config.chout['enc%d' % step], is_training=is_training, scope='enc%d' % step)
                encoders.append((encoder_pool, encoder))
            
            center = conv_block(encoders[-1][0], num_outputs=model_config.chout['center'], is_training=is_training, scope='center')
            
            encoders.reverse()
            for step in range(self.depth: 0: -1):
            input = center if step == 0 else encoders[step - 1][0]
            decoder = decoder_block(input, encoders[step][1], num_outputs=model_config.chout['dec%d' % step], is_training=is_training, scope='dec%d' % step)
            encoders.append((decoder, encoder))
    
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
