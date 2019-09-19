import tensorflow as tf
from models.retrieval.model import category_classifier, modality_classifier
import pdb
from models.retrieval import metric_loss_ops

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-6)

def category_loss(X1, X2, labels, num_categories):

    logits_image = category_classifier(X1, num_categories)
    logits_sent = category_classifier(X2, num_categories, reuse=True)
    labels_onehot = tf.one_hot(labels, num_categories)
    category_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_onehot, logits=logits_image) + \
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_onehot, logits=logits_sent)
    loss = tf.reduce_mean(category_loss)
    return loss

def triplet_loss(X1, X2, labels, margin = 1.0):
    
    X = tf.concat([X1, X2], axis = 0)
    all_labels = tf.concat([labels, labels], axis = 0)
    # loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(all_labels, X, margin)
    # return loss

    loss_img = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, X1, margin)
    loss_sen = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, X2, margin)
    loss_mod1 = metric_loss_ops.triplet_semihard_loss_modal(all_labels, X, margin)
    # loss_mod2 = metric_loss_ops.triplet_semihard_loss_modal(all_labels, tf.concat([X2, X1], axis = 0), margin)
    
    return 0.5*loss_mod1 + 0.1*loss_img + 0.5*loss_sen

def lifted_structured_loss(X1, X2, labels, margin = 1.0):
    X = tf.concat([X1, X2], axis = 0)
    all_labels = tf.concat([labels, labels], axis = 0)
    loss = tf.contrib.losses.metric_learning.lifted_struct_loss(all_labels, X, margin)
    return loss

# def modality_loss(X1, X2, lambda_, emb_dim, batch_size):

    # emb_v_class = modality_classifier(X1, lambda_, emb_dim)
    # emb_w_class = modality_classifier(X2, lambda_, emb_dim, reuse=True)
    # all_emb_v = tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], 1)
    # all_emb_w = tf.concat([tf.zeros([batch_size, 1]), tf.ones([batch_size, 1])], 1)
    # modality_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=emb_v_class, labels=all_emb_w) + \
        # tf.nn.softmax_cross_entropy_with_logits_v2(logits=emb_w_class, labels=all_emb_v)
    # loss = tf.reduce_mean(modality_loss)

    # return loss

def modality_loss(X1, X2, lambda_, emb_dim, batch_size, smooth = 0.02):

    logits_v_class = modality_classifier(X1, lambda_, emb_dim)
    logits_s_class = modality_classifier(X2, lambda_, emb_dim, reuse=True)
    
    d_labels_v = tf.ones_like(logits_v_class) * (1 - smooth)
    d_labels_s = tf.zeros_like(logits_s_class)

    d1_loss_v = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_v, logits=logits_v_class)
    d1_loss_s = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_labels_s, logits=logits_s_class)
    d1_loss = tf.reduce_mean(d1_loss_v + d1_loss_s)
    
    return d1_loss
