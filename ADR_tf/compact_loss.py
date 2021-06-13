import numpy as np 
import tensorflow as tf 
from utils_tf import tf_repeat, tf_shuffle, tf_roll
from tensorflow.keras import backend as K 

def softmax_cross_entropy_with_two_logits(logits=None, labels=None):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(labels), logits=logits)

def label_weight(y1, y2, scale, pairwise, one_hot=False): 
    # Scale should in (0., 1.) 
    if not one_hot:
        y1 = tf.one_hot(y1, 10, on_value=1.0, off_value=0.0, dtype=tf.float32) # [b, 10]
        y2 = tf.one_hot(y2, 10, on_value=1.0, off_value=0.0, dtype=tf.float32) # [b, 10]

    if pairwise: 
        y1_ = K.expand_dims(y1, axis=0) # [1, b, 10]
        y2_ = K.expand_dims(y2, axis=1) # [b, 1, 10]
    else: 
        y1_ = y1 
        y2_ = y2 

    d = tf.abs(y1_ - y2_) # [b, b, 10] if pairwise else [b, 10]
    d = tf.reduce_sum(d, axis=-1, keepdims=False) # [b, b], only two values [0., 2.]
    d = d / tf.reduce_max(d) # [0., 1.]
    d = scale - d # [scale, scale-1]
    d = d / scale # [1, 1 - 1/scale]

    return d 

def compact_loss(z, z_p, pairwise): 
    # z, z_p shape [b, k]

    if pairwise: 
        z = K.expand_dims(z, axis=0) # [1, b, k]
        z_p = K.expand_dims(z_p, axis=1) # [b, 1, k]

    loss = tf.reduce_mean(tf.abs(tf.stop_gradient(z) - z_p), axis=-1, keepdims=False) # [b, b] with pairwise, else [b, ]

    return loss 

def my_cross_entropy_with_two_logits(labels, logits): 
    # [b, b, d]
    [b, _, d] = labels.get_shape().as_list()

    _labels = K.reshape(labels, [b*b, d])
    _logits = K.reshape(logits, [b*b, d])

    loss = softmax_cross_entropy_with_two_logits(labels=_labels, logits=_logits)
    loss = K.reshape(loss, [b, b])

    return loss 


def smooth_loss(logits, logits_p, pairwise): 
    # logits, logits_p shape [b, d]
    if pairwise: 
        b, d = logits.get_shape().as_list()

        logits = K.expand_dims(logits, axis=0) # [1, b, d]
        logits_p = K.expand_dims(logits_p, axis=1) # [b, 1, d]

        logits = tf_repeat(logits, b, axis=0) # [b, b, d]
        logits_p = tf_repeat(logits_p, b, axis=1) # [b, b, d]

        loss = my_cross_entropy_with_two_logits(logits=logits_p, 
            labels=tf.stop_gradient(logits)) # [b, b]
        assert(loss.get_shape().as_list() == [b, b])
    else: 
        loss = softmax_cross_entropy_with_two_logits(logits=logits_p, 
            labels=tf.stop_gradient(logits)) # [b, ]

    return loss 

def global_loss(y, z, logits, y_p, z_p, logits_p, scale=0.99): 
    # Global scale should be smaller than local scale 
    assert(scale >= 0.9)
    c_loss = compact_loss(z, z_p, pairwise=True)
    s_loss = smooth_loss(logits, logits_p, pairwise=True)
    lw = label_weight(y1=y, y2=y_p, scale=scale, pairwise=True, one_hot=False)
    c_loss_weighted = tf.reduce_mean(tf.multiply(tf.stop_gradient(lw), c_loss))
    s_loss_weighted = tf.reduce_mean(tf.multiply(tf.stop_gradient(lw), s_loss))

    return c_loss_weighted, s_loss_weighted

def local_loss(y, z, logits, y_p, z_p, logits_p): 
    c_loss = compact_loss(z, z_p, pairwise=False)
    s_loss = smooth_loss(logits, logits_p, pairwise=False)
    return tf.reduce_mean(c_loss), tf.reduce_mean(s_loss)    



