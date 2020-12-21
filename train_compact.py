"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import sys
import shutil
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np

import cifar10_input
from pgd_attack import LinfPGDAttack
from trades import LinfTRADESAttack
from mysetting import * 
from pie.utils.utils_cm import writelog
from pie.utils.utils_cm import mkdir_p 
from utils import LogData, backup
from utils_grad import plot_grad_wrt_x_or_z, save_image

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()


from model import Model
model = Model(mode='train', mtype=args.model)

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)


def _add_loss(x, w, t): 
  if w != 0: 
    print('[**] add loss: {} with w={}'.format(t, w))
    return w*x 
  else: 
    return 0.
"""
  Calculate the loss with f() logit vector and p() probability vector
    mean_xent_perturb: cross entropy loss, 0.5*C(f(x), y) + 0.5*C(f(x_a), y)
    weight_decay_loss: weight decay loss 
    l_conf: confidence loss, H(f(x_a)) = C(f(x_a), p(x_a))
    l_vat: VAT loss, KL(p(x) || p(x_a))
    l_lc_com: Local compactness, dis(z, z_a)
    l_gb_com: Global compactness, dis(z_i, z_j) with label weighting 
    l_gb_vat: Global smoothness, KL(p(x_i) || p(x_j)) with label weighting 
  (The global smoothness is just for further investigation, set l_gb_smt=0.)
  Refer to the paper "Improving Adversarial Robustness by Enforcing Local and Global Compactness"
  for more detail.  
"""  

from compact_loss import softmax_cross_entropy_with_two_logits as softmax_xent_two
from compact_loss import local_loss 
from compact_loss import global_loss

l_conf = 1.0 *tf.reduce_mean(softmax_xent_two(model.logits_perturb, model.logits_perturb))

l_lc_com, l_vat = local_loss(model.y_input, model.z_clean, model.pre_softmax, 
  model.y_input, model.z_perturb, model.logits_perturb)

l_gb_com, l_gb_smt = global_loss(model.y_input, model.z_clean, model.pre_softmax, 
    model.y_input, model.z_perturb, model.logits_perturb, scale=0.99)

total_loss = 0 
total_loss += _add_loss(model.mean_xent, 0.5, 'cross_entropy') 
total_loss += _add_loss(model.mean_xent_perturb, 0.5, 'cross_entropy')
total_loss += _add_loss(model.weight_decay_loss, args.wdc, 'weight_decay_loss') 
total_loss += _add_loss(l_lc_com, args.lccomw, 'l_lc_com') 
total_loss += _add_loss(l_conf, args.confw, 'l_conf') 
total_loss += _add_loss(l_vat, args.vatw, 'l_vat') 
total_loss += _add_loss(l_gb_com, args.gbcomw, 'l_gb_com') 
total_loss += _add_loss(l_gb_smt, args.gbsmtw, 'l_gb_smt')

train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)


# Set up adversary
if args.perturb == 'pgd':
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])
elif args.perturb == 'trades': 
  attack = LinfTRADESAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         'trades')


# Setting up the Tensorboard and checkpoint outputs
model_name = '_'.join([t.format(v) for (t, v) in setup])
model_dir = os.path.join('models', model_name)
log_dir = os.path.join('logs', model_name)
log_file = log_dir + '/logfile.txt'
print("model_dir: {}".format(model_dir))
print("log_dir: {}".format(log_dir))
mkdir_p('models')
mkdir_p(model_dir)
mkdir_p('logs')
mkdir_p(log_dir)
mkdir_p(log_dir+'/images/')
mkdir_p(log_dir+'/codes/')
backup('./', log_dir+'/codes/')

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

def _eval_model(model, sess, raw_cifar): 
  nb = np.shape(raw_cifar.eval_data.xs)[0]//batch_size
  res = LogData()
  for t in range(nb): 
    xs = raw_cifar.eval_data.xs[t*batch_size:(t+1)*batch_size]
    ys = raw_cifar.eval_data.ys[t*batch_size:(t+1)*batch_size]
    xs_adv = attack.perturb(xs, ys, sess, useperturb=True)

    eval_dict = {model.x_input: xs, 
                model.x_perturb: xs_adv, 
                model.y_input: ys}
    acc, acc_p, ce, ce_p, dc, lc_com, conf, vat, gb_com, gb_smt = sess.run([model.accuracy, model.accuracy_p,
        model.mean_xent, model.mean_xent_perturb,
        model.weight_decay_loss, l_lc_com, l_conf, l_vat, l_gb_com, l_gb_smt], feed_dict=eval_dict)
    res.log(key='acc',    value=acc)
    res.log(key='acc_p',  value=acc_p)
    res.log(key='l_ce',    value=ce)
    res.log(key='l_per',   value=ce_p)
    res.log(key='l_wdc',     value=dc)
    res.log(key='l_lc_com',  value=lc_com)
    res.log(key='l_conf',   value=conf)
    res.log(key='l_vat',    value=vat)
    res.log(key='l_gb_com', value=gb_com)
    res.log(key='l_gb_smt', value=gb_smt)
  return res 

with tf.Session() as sess:

  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

  training_time = 0.0

  model_path = tf.train.latest_checkpoint(model_dir)

  if model_path is not None: 
    saver.restore(sess, model_path)
    print("Restored from {}".format(model_path))
  else: 
    print("Model not exists: {}".format(model_path))
    # quit()
    sess.run(tf.global_variables_initializer())

  logdata = LogData()

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)

    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess, useperturb=True)
    end = timer()
    training_time += end - start
    train_dict = {model.x_input: x_batch,
                model.x_perturb: x_batch_adv,
                model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      print('x_batch ', np.shape(x_batch), np.max(x_batch), np.min(x_batch))
      print('x_batch_adv ', np.shape(x_batch_adv), np.max(x_batch_adv), np.min(x_batch_adv))
      print('y_batch ', np.shape(y_batch), np.max(y_batch), np.min(y_batch))
      
      acc, acc_p, ce, ce_p, dc, lc_com, conf, vat, gb_com, gb_smt = sess.run([model.accuracy, model.accuracy_p,
        model.mean_xent, model.mean_xent_perturb,
        model.weight_decay_loss, l_lc_com, l_conf, l_vat, l_gb_com, l_gb_smt], feed_dict=train_dict)
      # writelog('Step {}:    ({})'.format(ii, datetime.now()), log_file)
      _writestr = [
        ('iter: {}',    ii), 
        ('time: ({})',   datetime.now()), 
        ('acc: {:.3f}',   acc*100), 
        ('acc_p: {:.3f}',   acc_p*100), 
        ('l_ce: {:.4f}',  ce), 
        ('l_per: {:.4f}',   ce_p),
        ('l_lc_com: {:.4f}',  lc_com),
        ('l_wdc: {:.4f}',   dc), 
        ('l_vat: {:.4f}',   vat), 
        ('l_conf: {:.4f}',  conf),
        ('l_gb_com: {:.4f}',  gb_com),
        ('l_gb_smt: {:.4f}',  gb_smt), 
      ]
      writestr = ' '.join([t.format(v) for (t, v) in _writestr])
      writelog(writestr, log_file)

      if ii % (num_output_steps*10) == 0:

        ev_res = _eval_model(model, sess, raw_cifar)
        _writestr = [
          ('eval-iter: {}',    ii), 
          ('eval-time: ({})',   datetime.now()), 
          ('acc: {:.3f}',   ev_res.mean('acc')*100), 
          ('acc_p: {:.3f}',   ev_res.mean('acc_p')*100), 
          ('l_ce: {:.4f}',  ev_res.mean('l_ce')), 
          ('l_per: {:.4f}',   ev_res.mean('l_per')),
          ('l_lc_com: {:.4f}',  ev_res.mean('l_lc_com')),
          ('l_wdc: {:.4f}',   ev_res.mean('l_wdc')), 
          ('l_vat: {:.4f}',   ev_res.mean('l_vat')), 
          ('l_conf: {:.4f}',  ev_res.mean('l_conf')),
          ('l_gb_com: {:.4f}',  ev_res.mean('l_gb_com')),
          ('l_gb_smt: {:.4f}',  ev_res.mean('l_gb_smt')), 
        ]
        writestr = ' '.join([t.format(v) for (t, v) in _writestr])
        writelog(writestr, log_file)

        logdata.log(key='iter', value=ii)
        logdata.lognplot2(key='acc', value=acc*100, value2=ev_res.mean('acc')*100, savepath=log_dir+'/images/acc.png')
        logdata.lognplot2(key='acc_p', value=acc_p*100, value2=ev_res.mean('acc_p')*100, savepath=log_dir+'/images/acc_p.png')
        logdata.lognplot2(key='l_ce', value=ce, value2=ev_res.mean('l_ce'), savepath=log_dir+'/images/l_ce.png')
        logdata.lognplot2(key='l_lc_com', value=lc_com, value2=ev_res.mean('l_lc_com'), savepath=log_dir+'/images/l_lc_com.png')
        logdata.lognplot2(key='l_wdc', value=dc, value2=ev_res.mean('l_wdc'), savepath=log_dir+'/images/l_wdc.png')
        logdata.lognplot2(key='l_conf', value=conf, value2=ev_res.mean('l_conf'), savepath=log_dir+'/images/l_conf.png')
        logdata.lognplot2(key='l_per', value=ce_p, value2=ev_res.mean('l_per'), savepath=log_dir+'/images/l_per.png')
        logdata.lognplot2(key='l_gb_com', value=gb_com, value2=ev_res.mean('l_gb_com'),savepath=log_dir+'/images/l_gb_com.png')
        logdata.lognplot2(key='l_gb_smt', value=gb_smt, value2=ev_res.mean('l_gb_smt'),savepath=log_dir+'/images/l_gb_smt.png')
        logdata.lognplot2(key='l_vat', value=vat, value2=ev_res.mean('l_vat'),savepath=log_dir+'/images/l_vat.png')


      if ii != 0:
        writelog('    {} examples per second'.format(
            num_output_steps * batch_size / training_time), log_file)
        training_time = 0.0

    if ii % 5000 == 0: 
      savepath = log_dir + '/images/iter_{}_attack_{}_'.format(ii, 'pgd')
      title = 'iter={},attack={}'.format(ii, 'pgd')      
      plot_grad_wrt_x_or_z(model, sess, x=x_batch_adv[0:1], y=y_batch[0:1], eps=config['epsilon'], savepath=savepath, title=title)

    if ii % 5000 == 0: 
      savepath = log_dir + '/images/iter_{}_attack_{}_examples'.format(ii, 'pgd')
      x_clean = np.zeros(shape=(32,32*5,3))
      x_adv = np.zeros(shape=(32,32*5,3))
      for t in range(5): 
        x_clean[:,32*t:32*t+32,:] = x_batch[t]
        x_adv[:,32*t:32*t+32,:] = x_batch_adv[t]
      save_image(x_clean, savepath+'_clean.png')
      save_image(x_adv, savepath+'_adv.png')


    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=train_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=train_dict)
    end = timer()
    training_time += end - start
