from utils.ops import *
from utils.utils import *

import warnings
warnings.filterwarnings('ignore')

class GanCls(object):
    def __init__(self, cfg, build_model=True):
        """
        Args:
          cfg: Config specifying all the parameters of the model.
        """

        self.name = 'GANL_CLS'

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.sample_num = cfg.TRAIN.SAMPLE_NUM

        self.output_size = cfg.MODEL.OUTPUT_SIZE

        self.z_dim = cfg.MODEL.Z_DIM
        self.embed_dim = cfg.MODEL.EMBED_DIM
        self.compressed_embed_dim = cfg.MODEL.COMPRESSED_EMBED_DIM

        self.gf_dim = cfg.MODEL.GF_DIM
        self.df_dim = cfg.MODEL.DF_DIM
        
        self.image_dims = [cfg.MODEL.IMAGE_SHAPE.H, cfg.MODEL.IMAGE_SHAPE.W, cfg.MODEL.IMAGE_SHAPE.D]

        self.w_init = tf.random_normal_initializer(stddev=0.02)
        self.batch_norm_init = {
            'gamma': tf.random_normal_initializer(1., 0.02),
        }

        if build_model:
            self.build_model()

    def build_model(self):
        # Define the input tensor by appending the batch size dimension to the image dimension
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='real_images')
        self.wrong_inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='wrong_images')
        self.phi_inputs = tf.placeholder(tf.float32, [self.batch_size] + [self.embed_dim], name='phi_inputs')

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        self.z_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.z_dim], name='z_sample')
        self.phi_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.embed_dim], name='phi_sample')

        self.G = self.generator(self.z, self.phi_inputs, reuse=False)
        self.D_synthetic, self.D_synthetic_logits = self.discriminator(self.G, self.phi_inputs, reuse=False)
        self.D_real_match, self.D_real_match_logits = self.discriminator(self.inputs, self.phi_inputs, reuse=True)
        self.D_real_mismatch, self.D_real_mismatch_logits = self.discriminator(self.wrong_inputs, self.phi_inputs,
                                                                               reuse=True)
        self.sampler = self.generator(self.z_sample, self.phi_sample, is_training=False, reuse=True)

    def discriminator(self, inputs, embed, is_training=True, reuse=False):
        s16 = self.output_size / 16

        with tf.variable_scope("d_net", reuse=reuse):
            net_ho = tf.layers.conv2d(inputs=inputs, filters=self.df_dim, kernel_size=(4, 4), strides=(2, 2),
                                      padding='same', activation=lambda l: tf.nn.leaky_relu(l, 0.2),
                                      kernel_initializer=self.w_init)
            net_h1 = tf.layers.conv2d(inputs=net_ho, filters=self.df_dim * 2, kernel_size=(4, 4), strides=(2, 2),
                                      padding='same', activation=None, kernel_initializer=self.w_init)
            net_h1 = batch_norm(net_h1, train=is_training, init=self.batch_norm_init,
                                act=lambda l: tf.nn.leaky_relu(l, 0.2))
            net_h2 = tf.layers.conv2d(inputs=net_h1, filters=self.df_dim * 4, kernel_size=(4, 4), strides=(2, 2),
                                      padding='same', activation=None, kernel_initializer=self.w_init)
            net_h2 = batch_norm(net_h2, train=is_training, init=self.batch_norm_init,
                                act=lambda l: tf.nn.leaky_relu(l, 0.2))
            net_h3 = tf.layers.conv2d(inputs=net_h2, filters=self.df_dim * 8, kernel_size=(4, 4), strides=(2, 2),
                                      padding='same', activation=None, kernel_initializer=self.w_init)
            net_h3 = batch_norm(net_h3, train=is_training, init=self.batch_norm_init,
                                act=None)
            # --------------------------------------------------------

            # Residual layer
            net = tf.layers.conv2d(inputs=net_h3, filters=self.df_dim * 2, kernel_size=(1, 1), strides=(1, 1),
                                   padding='valid', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=lambda l: tf.nn.leaky_relu(l, 0.2))
            net = tf.layers.conv2d(inputs=net, filters=self.df_dim * 2, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=lambda l: tf.nn.leaky_relu(l, 0.2))
            net = tf.layers.conv2d(inputs=net, filters=self.df_dim * 8, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=None)
            net_h4 = tf.add(net_h3, net)
            net_h4 = tf.nn.leaky_relu(net_h4, 0.2)
            # --------------------------------------------------------

            # Compress embeddings
            net_embed = tf.layers.dense(inputs=embed, units=self.compressed_embed_dim,
                                        activation=lambda l: tf.nn.leaky_relu(l, 0.2))

            # Append embeddings in depth
            net_embed = tf.expand_dims(tf.expand_dims(net_embed, 1), 1)
            net_embed = tf.tile(net_embed, [1, 4, 4, 1])
            net_h4_concat = tf.concat([net_h4, net_embed], axis=3)

            net_h4 = tf.layers.conv2d(inputs=net_h4_concat, filters=self.df_dim * 8, kernel_size=(1, 1), strides=(1, 1),
                                      padding='valid', activation=None, kernel_initializer=self.w_init)
            net_h4 = batch_norm(net_h4, train=is_training, init=self.batch_norm_init,
                                act=lambda l: tf.nn.leaky_relu(l, 0.2))

            net_logits = tf.layers.conv2d(inputs=net_h4, filters=1, kernel_size=(s16, s16), strides=(s16, s16),
                                          padding='valid', kernel_initializer=self.w_init)

            return tf.nn.sigmoid(net_logits), net_logits

    def generator(self, z, embed, is_training=True, reuse=False):
        s = self.output_size
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

        with tf.variable_scope("g_net", reuse=reuse):
            # Compress the embedding and append it to z
            net_embed = tf.layers.dense(inputs=embed, units=self.compressed_embed_dim, activation=None)

            # Concatenate the sampled embedding with the z vector
            net_input = tf.concat([z, net_embed], 1)
            net_h0 = tf.layers.dense(net_input, units=self.gf_dim * 8 * s16 * s16, activation=None,
                                     kernel_initializer=self.w_init)
            net_h0 = batch_norm(net_h0, train=is_training, init=self.batch_norm_init,
                                act=None)
            # --------------------------------------------------------
            net_h0 = tf.reshape(net_h0, [-1, 4, 4, self.gf_dim * 8])

            # Residual layer
            net = tf.layers.conv2d(inputs=net_h0, filters=self.gf_dim * 2, kernel_size=(1, 1), strides=(1, 1),
                                   padding='valid', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=tf.nn.relu)
            net = tf.layers.conv2d(inputs=net, filters=self.gf_dim * 2, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=tf.nn.relu)
            net = tf.layers.conv2d(inputs=net, filters=self.gf_dim * 8, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=None)
            net_h1 = tf.add(net_h0, net)
            net_h1 = tf.nn.relu(net_h1)
            # --------------------------------------------------------

            net_h2 = tf.layers.conv2d_transpose(net_h1, filters=self.gf_dim * 4, kernel_size=(4, 4), strides=(2, 2),
                                                padding='same', activation=None, kernel_initializer=self.w_init)
            net_h2 = tf.layers.conv2d(inputs=net_h2, filters=self.gf_dim * 4, kernel_size=(3, 3), strides=(1, 1),
                                      padding='same', activation=None, kernel_initializer=self.w_init)
            net_h2 = batch_norm(net_h2, train=is_training, init=self.batch_norm_init,
                                act=None)
            # --------------------------------------------------------

            # Residual layer
            net = tf.layers.conv2d(inputs=net_h2, filters=self.gf_dim, kernel_size=(1, 1), strides=(1, 1),
                                   padding='valid', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=tf.nn.relu)
            net = tf.layers.conv2d(inputs=net, filters=self.gf_dim, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=tf.nn.relu)
            net = tf.layers.conv2d(inputs=net, filters=self.gf_dim * 4, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=None, kernel_initializer=self.w_init)
            net = batch_norm(net, train=is_training, init=self.batch_norm_init,
                             act=None)
            net_h3 = tf.add(net_h2, net)
            net_h3 = tf.nn.relu(net_h3)
            # --------------------------------------------------------

            net_h4 = tf.layers.conv2d_transpose(net_h3, filters=self.gf_dim * 2, kernel_size=(4, 4), strides=(2, 2),
                                                padding='same', activation=None, kernel_initializer=self.w_init)
            net_h4 = tf.layers.conv2d(inputs=net_h4, filters=self.gf_dim * 2, kernel_size=(3, 3), strides=(1, 1),
                                      padding='same', activation=None, kernel_initializer=self.w_init)
            net_h4 = batch_norm(net_h4, train=is_training, init=self.batch_norm_init,
                                act=tf.nn.relu)

            net_h5 = tf.layers.conv2d_transpose(net_h4, filters=self.gf_dim, kernel_size=(4, 4), strides=(2, 2),
                                                padding='same', activation=None, kernel_initializer=self.w_init)
            net_h5 = tf.layers.conv2d(inputs=net_h5, filters=self.gf_dim, kernel_size=(3, 3), strides=(1, 1),
                                      padding='same', activation=None, kernel_initializer=self.w_init)
            net_h5 = batch_norm(net_h5, train=is_training, init=self.batch_norm_init,
                                act=tf.nn.relu)

            net_logits = tf.layers.conv2d_transpose(net_h5, filters=self.image_dims[-1], kernel_size=(4, 4),
                                                    strides=(2, 2), padding='same', activation=None,
                                                    kernel_initializer=self.w_init)
            net_logits = tf.layers.conv2d(inputs=net_logits, filters=self.image_dims[-1], kernel_size=(3, 3),
                                          strides=(1, 1), padding='same', activation=None,
                                          kernel_initializer=self.w_init)

            net_output = tf.nn.tanh(net_logits)
            return net_output
