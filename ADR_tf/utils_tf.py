"""Utility functions for writing TensorFlow code"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import math
import os
import time
import warnings

import numpy as np
import six
from six.moves import xrange
import tensorflow as tf

def tf_shuffle(x): 
  return tf.gather(x, tf.random_shuffle(tf.range(tf.shape(x)[0])))

def tf_roll(x, shift=1, axis=0): 
  # return tf.roll(x, shift=shift, axis=axis)
  assert(axis==0)
  nb = x.get_shape().as_list()[0]
  y = tf.concat([x[shift:], x[0:shift]], axis=axis)
  return y 

def tf_repeat(x, n, axis=0): 
  x_shape = x.get_shape().as_list() # [b, d1, d2]
  multiples = len(x_shape) * [1]
  if x_shape[axis] is not None: 
    multiples[axis] = n

  x_r = tf.tile(x, multiples=multiples)
  return x_r 

def l2_batch_normalize(x, epsilon=1e-12, scope=None, axis=None, keepdims=None):
  """
  Helper function to normalize a batch of vectors.
  :param x: the input placeholder
  :param epsilon: stabilizes division
  :return: the batch of l2 normalized vector
  """
  with tf.name_scope(scope, "l2_batch_normalize") as name_scope:
    x_shape = tf.shape(x)
    x = tf.contrib.layers.flatten(x)
    x /= (epsilon + tf.reduce_max(tf.abs(x), 1, keepdims=True))
    square_sum = tf.reduce_sum(tf.square(x), 1, keepdims=True)
    x_inv_norm = tf.rsqrt(np.sqrt(epsilon) + square_sum)
    x_norm = tf.multiply(x, x_inv_norm)
    return tf.reshape(x_norm, x_shape, name_scope)

def kl_with_logits(p_logits, q_logits, scope=None,
                   loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
  """Helper function to compute kl-divergence KL(p || q)
  """
  with tf.name_scope(scope, "kl_divergence") as name:
    p = tf.nn.softmax(p_logits)
    p_log = tf.nn.log_softmax(p_logits)
    q_log = tf.nn.log_softmax(q_logits)
    loss = tf.reduce_mean(tf.reduce_sum(p * (p_log - q_log), axis=1),
                       name=name)
    tf.losses.add_loss(loss, loss_collection)
    return loss

def add_eta(x, ord, eps, clip_value_min=None, clip_value_max=None, seed=None): 
  eta = random_lp_vector(tf.shape(x), ord=ord, eps=eps, 
    dtype=tf.float32, seed=seed)
  eta = clip_eta(eta, ord, eps)
  adv_x = x + eta 
  if clip_value_min is not None or clip_value_max is not None: 
    adv_x = clip_by_value(adv_x, clip_value_min, clip_value_max)
  return adv_x

def add_eta_np(x, eta, ord, eps, clip_value_min=None, clip_value_max=None, seed=None): 
  eta = clip_eta(eta, ord, eps)
  adv_x = x + eta 
  if clip_value_min is not None or clip_value_max is not None: 
    adv_x = clip_by_value(adv_x, clip_value_min, clip_value_max)
  return adv_x


def clip_eta(eta, ord, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.ord norm ball
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')
  reduc_ind = list(xrange(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    eta = clip_by_value(eta, -eps, eps)
  elif ord == 1:
    # Implements a projection algorithm onto the l1-ball from
    # (Duchi et al. 2008) that runs in time O(d*log(d)) where d is the
    # input dimension.
    # Paper link (Duchi et al. 2008): https://dl.acm.org/citation.cfm?id=1390191

    eps = tf.cast(eps, eta.dtype)

    dim = tf.reduce_prod(tf.shape(eta)[1:])
    eta_flat = tf.reshape(eta, (-1, dim))
    abs_eta = tf.abs(eta_flat)

    if 'sort' in dir(tf):
      mu = -tf.sort(-abs_eta, axis=-1)
    else:
      # `tf.sort` is only available in TF 1.13 onwards
      mu = tf.nn.top_k(abs_eta, k=dim, sorted=True)[0]
    cumsums = tf.cumsum(mu, axis=-1)
    js = tf.cast(tf.divide(1, tf.range(1, dim + 1)), eta.dtype)
    t = tf.cast(tf.greater(mu - js * (cumsums - eps), 0), eta.dtype)

    rho = tf.argmax(t * cumsums, axis=-1)
    rho_val = tf.reduce_max(t * cumsums, axis=-1)
    theta = tf.divide(rho_val - eps, tf.cast(1 + rho, eta.dtype))

    eta_sgn = tf.sign(eta_flat)
    eta_proj = eta_sgn * tf.maximum(abs_eta - theta[:, tf.newaxis], 0)
    eta_proj = tf.reshape(eta_proj, tf.shape(eta))

    norm = tf.reduce_sum(tf.abs(eta), reduc_ind)
    eta = tf.where(tf.greater(norm, eps), eta_proj, eta)

  elif ord == 2:
    # avoid_zero_div must go inside sqrt to avoid a divide by zero
    # in the gradient through this operation
    norm = tf.sqrt(tf.maximum(avoid_zero_div,
                              tf.reduce_sum(tf.square(eta),
                                         reduc_ind,
                                         keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the
    # surface of the ball
    factor = tf.minimum(1., div(eps, norm))
    eta = eta * factor
  return eta


def zero_out_clipped_grads(grad, x, clip_min, clip_max):
  """
  Helper function to erase entries in the gradient where the update would be
  clipped.
  :param grad: The gradient
  :param x: The current input
  :param clip_min: Minimum input component value
  :param clip_max: Maximum input component value
  """
  signed_grad = tf.sign(grad)

  # Find input components that lie at the boundary of the input range, and
  # where the gradient points in the wrong direction.
  clip_low = tf.logical_and(tf.less_equal(x, tf.cast(clip_min, x.dtype)),
                            tf.less(signed_grad, 0))
  clip_high = tf.logical_and(tf.greater_equal(x, tf.cast(clip_max, x.dtype)),
                             tf.greater(signed_grad, 0))
  clip = tf.logical_or(clip_low, clip_high)
  grad = tf.where(clip, mul(grad, 0), grad)

  return grad


def random_exponential(shape, rate=1.0, dtype=tf.float32, seed=None):
  """
  Helper function to sample from the exponential distribution, which is not
  included in core TensorFlow.
  """
  return tf.random_gamma(shape, alpha=1, beta=1. / rate, dtype=dtype, seed=seed)


def random_laplace(shape, loc=0.0, scale=1.0, dtype=tf.float32, seed=None):
  """
  Helper function to sample from the Laplace distribution, which is not
  included in core TensorFlow.
  """
  z1 = random_exponential(shape, loc, dtype=dtype, seed=seed)
  z2 = random_exponential(shape, scale, dtype=dtype, seed=seed)
  return z1 - z2


def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
  """
  Helper function to generate uniformly random vectors from a norm ball of
  radius epsilon.
  :param shape: Output shape of the random sample. The shape is expected to be
                of the form `(n, d1, d2, ..., dn)` where `n` is the number of
                i.i.d. samples that will be drawn from a norm ball of dimension
                `d1*d1*...*dn`.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, radius of the norm ball.
  """
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')

  if ord == np.inf:
    r = tf.random_uniform(shape, -eps, eps, dtype=dtype, seed=seed)
  else:

    # For ord=1 and ord=2, we use the generic technique from
    # (Calafiore et al. 1998) to sample uniformly from a norm ball.
    # Paper link (Calafiore et al. 1998):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=758215&tag=1
    # We first sample from the surface of the norm ball, and then scale by
    # a factor `w^(1/d)` where `w~U[0,1]` is a standard uniform random variable
    # and `d` is the dimension of the ball. In high dimensions, this is roughly
    # equivalent to sampling from the surface of the ball.

    dim = tf.reduce_prod(shape[1:])

    if ord == 1:
      x = random_laplace((shape[0], dim), loc=1.0, scale=1.0, dtype=dtype,
                         seed=seed)
      norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
    elif ord == 2:
      x = tf.random_normal((shape[0], dim), dtype=dtype, seed=seed)
      norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
    else:
      raise ValueError('ord must be np.inf, 1, or 2.')

    w = tf.pow(tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed),
               1.0 / tf.cast(dim, dtype))
    r = eps * tf.reshape(w * x / norm, shape)

  return r

def random_lp_vector_np(shape, ord, eps, seed=None):
  np.random.seed(seed)

  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')

  if ord == np.inf:
    r = np.random.uniform(size=shape, low=-eps, high=eps)
  else:

    # For ord=1 and ord=2, we use the generic technique from
    # (Calafiore et al. 1998) to sample uniformly from a norm ball.
    # Paper link (Calafiore et al. 1998):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=758215&tag=1
    # We first sample from the surface of the norm ball, and then scale by
    # a factor `w^(1/d)` where `w~U[0,1]` is a standard uniform random variable
    # and `d` is the dimension of the ball. In high dimensions, this is roughly
    # equivalent to sampling from the surface of the ball.

    dim = np.prod(shape[1:])

    if ord == 1:
      x = np.random.laplace(size=(shape[0], dim), loc=1.0, scale=1.0)
      norm = np.sum(np.abs(x), axis=-1, keepdims=True)
    elif ord == 2:
      x = np.random.normal(size=(shape[0], dim))
      norm = np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True))
    else:
      raise ValueError('ord must be np.inf, 1, or 2.')

    w = np.power(np.random.uniform(size=(shape[0], 1)), 1.0 / dim)
    r = eps * np.reshape(w * x / norm, shape)

  return r

def clip_by_value(t, clip_value_min, clip_value_max, name=None):
  """
  A wrapper for clip_by_value that casts the clipping range if needed.
  """
  def cast_clip(clip):
    """
    Cast clipping range argument if needed.
    """
    if t.dtype in (tf.float32, tf.float64):
      if hasattr(clip, 'dtype'):
        # Convert to tf dtype in case this is a numpy dtype
        clip_dtype = tf.as_dtype(clip.dtype)
        if clip_dtype != t.dtype:
          return tf.cast(clip, t.dtype)
    return clip

  clip_value_min = cast_clip(clip_value_min)
  clip_value_max = cast_clip(clip_value_max)

  return tf.clip_by_value(t, clip_value_min, clip_value_max, name)


def mul(a, b):
  """
  A wrapper around tf multiplication that does more automatic casting of
  the input.
  """
  def multiply(a, b):
    """Multiplication"""
    return a * b
  return op_with_scalar_cast(a, b, multiply)

def div(a, b):
  """
  A wrapper around tf division that does more automatic casting of
  the input.
  """
  def divide(a, b):
    """Division"""
    return a / b
  return op_with_scalar_cast(a, b, divide)

def op_with_scalar_cast(a, b, f):
  """
  Builds the graph to compute f(a, b).
  If only one of the two arguments is a scalar and the operation would
  cause a type error without casting, casts the scalar to match the
  tensor.
  :param a: a tf-compatible array or scalar
  :param b: a tf-compatible array or scalar
  """

  try:
    return f(a, b)
  except (TypeError, ValueError):
    pass

  def is_scalar(x):
    """Return True if `x` is a scalar"""
    if hasattr(x, "get_shape"):
      shape = x.get_shape()
      return shape.ndims == 0
    if hasattr(x, "ndim"):
      return x.ndim == 0
    assert isinstance(x, (int, float))
    return True

  a_scalar = is_scalar(a)
  b_scalar = is_scalar(b)

  if a_scalar and b_scalar:
    raise TypeError("Trying to apply " + str(f) + " with mixed types")

  if a_scalar and not b_scalar:
    a = tf.cast(a, b.dtype)

  if b_scalar and not a_scalar:
    b = tf.cast(b, a.dtype)

  return f(a, b)

def assert_less_equal(*args, **kwargs):
  """
  Wrapper for tf.assert_less_equal
  Overrides tf.device so that the assert always goes on CPU.
  The unwrapped version raises an exception if used with tf.device("/GPU:x").
  """
  with tf.device("/CPU:0"):
    return tf.assert_less_equal(*args, **kwargs)

def assert_greater_equal(*args, **kwargs):
  """
  Wrapper for tf.assert_greater_equal.
  Overrides tf.device so that the assert always goes on CPU.
  The unwrapped version raises an exception if used with tf.device("/GPU:x").
  """
  with tf.device("/CPU:0"):
    return tf.assert_greater_equal(*args, **kwargs)

def assert_equal(*args, **kwargs):
  """
  Wrapper for tf.assert_equal.
  Overrides tf.device so that the assert always goes on CPU.
  The unwrapped version raises an exception if used with tf.device("/GPU:x").
  """
  with tf.device("/CPU:0"):
    return tf.assert_equal(*args, **kwargs)