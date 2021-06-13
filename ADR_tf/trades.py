"""
Tensorflow reimplementation for TRADES Attack.
References: 
- Original Pytorch Implementation 
https://github.com/yaodongyu/TRADES/blob/master/trades.py
- Madry PGD Attack 
https://github.com/MadryLab/cifar10_challenge/blob/master/pgd_attack.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import cifar10_input
from loss import logsoftmax

class LinfTRADESAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func, targeted=False):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start

    logits = model.pre_softmax
    adv_logits = model.logits_perturb

    assert(loss_func == 'trades')
    q = tf.nn.softmax(adv_logits)
    qlogq = tf.reduce_sum(q * logsoftmax(adv_logits), 1)
    qlogp = tf.reduce_sum(q * logsoftmax(logits), 1)
    loss = qlogq - qlogp

    if targeted: 
      loss = -loss 
      
    self.grad = tf.gradients(loss, model.x_perturb)[0]

  def perturb(self, x_nat, y, sess, useperturb=False):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = x_nat.astype(np.float)

    for i in range(self.num_steps):
      assert(useperturb==True)
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x_nat,
                                          self.model.x_perturb: x,
                                          self.model.y_input: y})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x

