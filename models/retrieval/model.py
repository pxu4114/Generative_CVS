import tensorflow as tf
from tensorflow.contrib import slim
# from flip_gradient import flip_gradient
tf.set_random_seed(999)

def visual_feature_embed(X, emb_dim, is_training=True, reuse=False,dropout_ratio=0.5):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = tf.nn.tanh(slim.dropout(slim.fully_connected(X, emb_dim, scope='vf_fc_0'),keep_prob=dropout_ratio, is_training=is_training))
        net = tf.nn.tanh(slim.dropout(slim.fully_connected(net, emb_dim, scope='vf_fc_1'),keep_prob=dropout_ratio, is_training=is_training))
        net = slim.fully_connected(net, emb_dim, scope='vf_fc_2')
        # emb = tf.nn.l2_normalize(net, 1, epsilon=1e-10)
    return net

def sent_feature_embed(L, emb_dim, is_training=True, reuse=False,dropout_ratio=0.5):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = tf.nn.tanh(slim.dropout(slim.fully_connected(L, emb_dim, scope='sf_fc_0'),keep_prob=dropout_ratio, is_training=is_training))
        net = tf.nn.tanh(slim.dropout(slim.fully_connected(net, emb_dim, scope='sf_fc_1'),keep_prob=dropout_ratio, is_training=is_training))
        net = slim.fully_connected(net, emb_dim, scope='sf_fc_2')
        # emb = tf.nn.l2_normalize(net, 1, epsilon=1e-10)
    return net

def shared_embed(X, emb_dim, is_training=True, reuse=False,dropout_ratio=0.5):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse, trainable=is_training):
        net = tf.nn.tanh(slim.dropout(slim.fully_connected(X, 512, scope='se_fc_0'),keep_prob=dropout_ratio, is_training=is_training))
        net = slim.fully_connected(net, emb_dim, scope='se_fc_1')
        net = tf.nn.l2_normalize(net, 1, epsilon=1e-10)
    return net
    
def category_classifier(X, num_categories, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = tf.nn.tanh(slim.fully_connected(X, 256, scope='cc_fc_0'))
        net = slim.fully_connected(net, num_categories, scope='cc_fc_1')
    return net

def modality_classifier(E, lambda_, emb_dim, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        # E = flip_gradient(E, lambda_)
        net = tf.nn.tanh(slim.fully_connected(E, emb_dim/2, scope='mc_fc_0'))
        net = tf.nn.tanh(slim.fully_connected(net, emb_dim/4, scope='mc_fc_1'))
        net = slim.fully_connected(net, 1, scope='mc_fc_2')
    return net

def aligned_attention(X1, X2, emb_dim, is_training=True, reuse=tf.AUTO_REUSE, skip=True):
    with tf.variable_scope('att', reuse=tf.AUTO_REUSE):

        image_W = tf.get_variable('att_img', [emb_dim, emb_dim], trainable = is_training)
        sent_W = tf.get_variable('att_sent', [emb_dim, emb_dim], trainable = is_training)
        att_W = tf.get_variable('att_enc_W', [emb_dim, emb_dim], trainable = is_training)
        att_b = tf.get_variable('att_enc_b', [emb_dim], trainable = is_training)
        image_sent_W = tf.get_variable('att_image_sent', [emb_dim, emb_dim], trainable = is_training)
        
        # e_it = tf.matmul(tf.tanh(tf.nn.bias_add(tf.matmul(X1,image_W)+tf.matmul(X2,sent_W)+tf.multiply(X1,X2),att_b)),att_W)
        e_it = tf.matmul(tf.tanh(tf.nn.bias_add(tf.matmul(X1,image_W)+tf.matmul(X2,sent_W)+tf.matmul(tf.multiply(X1,X2), image_sent_W),att_b)),att_W)
        # e_it = tf.tanh(tf.nn.bias_add(tf.matmul(X1,image_W)+tf.matmul(X2,sent_W)+tf.matmul(tf.multiply(X1,X2), image_sent_W),att_b))
        # e_it = tf.matmul(tf.multiply(X1,X2), image_sent_W)
        
        alpha_it = tf.nn.softmax(e_it)
        
        if skip:
            X1_att = X1 + tf.multiply(alpha_it, X1)
            X2_att = X2 + tf.multiply(alpha_it, X2)
        else:
            X1_att = tf.multiply(alpha_it, X1)
            X2_att = tf.multiply(alpha_it, X2)            
        # if not is_training:
            # X1_att = X1
            # X2_att = X2
        X1_att = tf.nn.l2_normalize(X1_att, 1, epsilon=1e-10)
        X2_att = tf.nn.l2_normalize(X2_att, 1, epsilon=1e-10)

        return X1_att, X2_att, alpha_it


# def _embedding_model(input_tensor, \
                     # num_layers, \
                     # embedding_dimension):
    # return slim.repeat(input_tensor, num_layers, slim.fully_connected, embedding_dimension)

# def model(image, sent, embedding_size, number_layer = 2, activation_fn = tf.nn.tanh):
     # with slim.arg_scope([slim.fully_connected], \
                         # activation_fn=activation_fn, \
                         # weights_regularizer=slim.l2_regularizer(0.001)):
         # image_embedding = _embedding_model(image, number_layer, embedding_size)
         # caption_embedding = _embedding_model(sent, number_layer, embedding_size)
     # return image_embedding, caption_embedding

# def _embedding_model(X, embedding_size):
    # with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=False):
        # net = tf.nn.tanh(slim.fully_connected(X, 1024))
        # net = tf.nn.tanh(slim.fully_connected(net, 512))
        # net = tf.nn.tanh(slim.fully_connected(net, embedding_size))
    # return net

# def model(image, sent, embedding_size):
    # image_emb = _embedding_model(image, embedding_size)
    # caption_emb = _embedding_model(sent, embedding_size)
    # image_embedding = tf.nn.l2_normalize(image_emb, 1, epsilon=1e-10)
    # caption_embedding = tf.nn.l2_normalize(caption_emb, 1, epsilon=1e-10)
    # return image_embedding, caption_embedding
    