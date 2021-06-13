import numpy as np 
import tensorflow as tf 
from utils_tf import tf_repeat, tf_shuffle, tf_roll
from tensorflow.keras import backend as K 

def softmax_cross_entropy_with_two_logits(logits=None, labels=None):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(labels), logits=logits)

def cross_entropy(logits, labels, axis=-1, keepdims=False):
	# del keepdims 
	# return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=axis))
    p = labels 
    q = tf.nn.softmax(logits)
    loss = -tf.reduce_sum(tf.multiply(p, tf.log(q+1e-9)), axis=-1, keepdims=True)
    print('cross_entropy: ', loss.get_shape().as_list())
    loss = tf.reduce_mean(loss)
    print('cross_entropy: ', loss.get_shape().as_list())
    return loss 

def tf_cross_entropy(p, q, axis=-1, keepdims=False): 
	loss = -tf.reduce_sum(tf.multiply(p, tf.log(q+1e-12)), axis=axis, keepdims=keepdims)
	return loss 

def tf_kl_divergence(p, q, axis=-1, keepdims=False, shuffle=False):
    if shuffle: 
        q = tf_shuffle(q) 
    return tf_cross_entropy(p, q, axis, keepdims) - tf_cross_entropy(p, p, axis, keepdims)

def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm
  

def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp

def entropy_y_x(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))
