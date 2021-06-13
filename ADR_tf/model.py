# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Model(object):
  """ResNet model."""

  def __init__(self, mode, mtype):
    """ResNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
      mtype: Model type
              "wideresnet" with filters = [16, 160, 320, 640] 
              or "nonewide" with filters = [16, 16, 32, 64]
    """
    self.mode = mode
    self.mtype = mtype

    batch_size = 128 if self.mode == 'train' else None

    self.x_input = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])

    self.x_perturb = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])

    self.x_eval = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    self.y_eval = tf.placeholder(tf.int64, shape=[None,])

    self.y_input = tf.placeholder(tf.int64, shape=[batch_size,])

    self.pre_softmax, self.z_clean = self._build_model(self.x_input)
    self.logits_perturb, self.z_perturb = self._build_model(self.x_perturb)
    self.logits_eval, self.z_eval = self._build_model(self.x_eval)

    self.weight_decay_loss = self._decay()

    self.mean_xent, self.xent, self.accuracy, self.num_correct = self._get_loss(self.pre_softmax, self.y_input)
    self.mean_xent_perturb, self.xent_perturb, self.accuracy_p, _ = self._get_loss(self.logits_perturb, self.y_input)
    self.mean_xent_eval, self.xent_eval, _, _ = self._get_loss(self.logits_eval, self.y_eval)

    self.X_grad = tf.gradients(self.xent_eval, [self.x_eval])[0]
    self.z_grad = tf.gradients(self.xent_eval, [self.z_eval])[0]
    self.X_cls = tf.argmax(self.logits_eval, axis=1)

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self, x_input):
    assert self.mode == 'train' or self.mode == 'eval'

    input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                            x_input)

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))

      print('init_conv', x.get_shape().as_list())

      strides = [1, 2, 2]
      activate_before_residual = [True, False, False]
      res_func = self._residual

      # wide residual network (https://arxiv.org/abs/1605.07146v1)
      # use filters = [16, 16, 32, 64] for a non-wide version
      # filters = [16, 160, 320, 640]
      if self.mtype == 'nonwide':
        filters = [16, 16, 32, 64]
        z_dim = 64
      elif self.mtype == 'wideresnet': 
        filters = [16, 160, 320, 640]
        z_dim = 640

      # Update hps.num_residual_units to 9

      with tf.variable_scope('unit_1_0'):
        x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                     activate_before_residual[0])
      print('unit_1_0', x.get_shape().as_list())
      for i in range(1, 5):
        with tf.variable_scope('unit_1_%d' % i):
          x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

      with tf.variable_scope('unit_2_0'):
        x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                     activate_before_residual[1])
      print('unit_2_0', x.get_shape().as_list())
      for i in range(1, 5):
        with tf.variable_scope('unit_2_%d' % i):
          x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

      with tf.variable_scope('unit_3_0'):
        x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                     activate_before_residual[2])
      print('unit_3_0', x.get_shape().as_list())

      for i in range(1, 5):
        with tf.variable_scope('unit_3_%d' % i):
          x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

      with tf.variable_scope('unit_last'):
        x = self._batch_norm('final_bn', x)
        x = self._relu(x, 0.1)
        x = self._global_avg_pool(x)

      print('unit_last', x.get_shape().as_list())
      z = tf.reshape(x, [tf.shape(x)[0], np.prod(x.get_shape().as_list()[1:])])

      with tf.variable_scope('logits'):
        logits = self._fully_connected(x, 10, name='output')

      return logits, z 

  def _get_loss(self, logits, y_input):
    predictions = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predictions, y_input)
    num_correct = tf.reduce_sum(
        tf.cast(correct_prediction, tf.int64))
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=y_input)
    xent = tf.reduce_sum(y_xent)
    mean_xent = tf.reduce_mean(y_xent)

    return mean_xent, xent, accuracy, num_correct

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim, name='fc'):
    """FullyConnected layer for final output."""
    # num_non_batch_dimensions = len(x.shape)
    # prod_non_batch_dimensions = 1
    # for ii in range(num_non_batch_dimensions - 1):
    #   prod_non_batch_dimensions *= int(x.shape[ii + 1])

    prod_non_batch_dimensions = np.prod(x.get_shape().as_list()[1:])
    # x = tf.reshape(x, [tf.shape(x)[0], -1])
    print('fc x: ', x.get_shape().as_list())
    w = tf.get_variable(
        name+'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    print('fc w: ', w.get_shape().as_list())
    b = tf.get_variable(name+'biases', [out_dim],
                        initializer=tf.constant_initializer())
    print('fc b: ', b.get_shape().as_list())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])



