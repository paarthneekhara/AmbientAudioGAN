import os

import numpy as np
import tensorflow as tf

from AudioModel.conv1d import WaveEncoderFactor256, WaveDecoderFactor256
from AudioModel.model import Model, Modes
import measurement

class WaveAE(Model):
  subseq_len = 16384
  audio_fs = 16000
  batchnorm = False
  objective = 'l2'
  phaseshuffle_rad = 0
  zdim = 100
  wgangp_lambda = 10
  wgangp_nupdates = 5
  gan_strategy = 'wgangp'
  train_batch_size = 64
  alpha = 100.0
  eval_batch_size = 1
  dim = 64
  kernel_len = 5
  stride = 4

  def __init__(self, mode, *args, **kwargs):
    super().__init__(mode, *args, **kwargs)
    if self.mode == Modes.EVAL:
      self.best_l2 = None


  def build_generator(self, x):
    try:
      batch_size = int(x.get_shape()[0])
    except:
      batch_size = tf.shape(x)[0]

    training = self.mode == Modes.TRAIN

    # with tf.variable_scope('Gen'):
    with tf.variable_scope('E'):
      enc = WaveEncoderFactor256(batchnorm=self.batchnorm)
      self.E_x = E_x = enc(x, training=training)

    z = tf.random_uniform([batch_size, self.zdim], -1, 1, dtype=tf.float32)
    with tf.variable_scope('z_project'):
      z_proj = tf.layers.dense(z, 64 * 64)
      z_proj = tf.reshape(z_proj, [batch_size, 64, 1, 64])
      z_proj = tf.nn.tanh(z_proj)
    E_x_concat = tf.concat([E_x, z_proj], axis = -1)
    print("E_x concat", E_x_concat)
    


    with tf.variable_scope('D'):
      dec = WaveDecoderFactor256(batchnorm=self.batchnorm)
      self.D_E_x = D_E_x = dec(E_x_concat, training=training)
    print("Decoded")
    print(self.D_E_x)

    return E_x, D_E_x

  def build_discriminator(self, x):
    conv1d = lambda x, n: tf.layers.conv2d(
        x,
        n,
        (self.kernel_len, 1),
        strides=(self.stride, 1),
        padding='same')

    def lrelu(inputs, alpha=0.2):
      return tf.maximum(alpha * inputs, inputs)

    def apply_phaseshuffle(x, rad, pad_type='reflect'):
      if rad == 0:
        return x

      b, x_len, _, nch = x.get_shape().as_list()

      phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
      pad_l = tf.maximum(phase, 0)
      pad_r = tf.maximum(-phase, 0)
      phase_start = pad_r
      x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0], [0, 0]], mode=pad_type)

      x = x[:, phase_start:phase_start+x_len]
      x.set_shape([b, x_len, 1, nch])

      return x

    batch_size = tf.shape(x)[0]

    phaseshuffle = lambda x: apply_phaseshuffle(x, self.phaseshuffle_rad)

    # Layer 0
    # [16384, 1] -> [4096, 64]
    output = x
    with tf.variable_scope('downconv_0'):
      output = conv1d(output, self.dim)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 1
    # [4096, 64] -> [1024, 128]
    with tf.variable_scope('downconv_1'):
      output = conv1d(output, self.dim * 2)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 2
    # [1024, 128] -> [256, 256]
    with tf.variable_scope('downconv_2'):
      output = conv1d(output, self.dim * 4)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 3
    # [256, 256] -> [64, 512]
    with tf.variable_scope('downconv_3'):
      output = conv1d(output, self.dim * 8)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 4
    # [64, 512] -> [16, 1024]
    with tf.variable_scope('downconv_4'):
      output = conv1d(output, self.dim * 16)
    output = lrelu(output)

    # Flatten
    output = tf.reshape(output, [batch_size, 4 * 4 * self.dim * 16])

    # Connect to single logit
    with tf.variable_scope('output'):
      output = tf.layers.dense(output, 1)[:, 0]

    return output

  def __call__(self, x):
    try:
      batch_size = int(x.get_shape()[0])
    except:
      batch_size = tf.shape(x)[0]

    training = self.mode == Modes.TRAIN

    # making noisy signal
    clean_audio = x
    _, x = measurement.block_patch(x[:,:,0,0], patch_size = 4000)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)
    

    with tf.variable_scope('Gen'):
      E_x, D_E_x = self.build_generator(x)

    # zeros where input is 0, one else where
    input_mask = tf.cast( tf.greater(tf.abs(x), tf.zeros_like(x)), dtype = tf.float32 )
    pads, measured_audio = measurement.block_patch(D_E_x[:,:,0,0], patch_size = 4000)
    print (measured_audio)
    measured_expanded = tf.expand_dims(tf.expand_dims(measured_audio, -1), -1)
    print(measured_expanded)

    with tf.name_scope('D_x'), tf.variable_scope('Disc'):
      D_x = self.build_discriminator(x)
    with tf.name_scope('D_g'), tf.variable_scope('Disc', reuse=True):
      D_g = self.build_discriminator(measured_expanded)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

    

    if self.gan_strategy == 'dcgan':
      D_G_z = D_g
      D_x = D_x
      fake = tf.zeros([batch_size], dtype=tf.float32)
      real = tf.ones([batch_size], dtype=tf.float32)

      G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_G_z,
        labels=real
      ))

      D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_G_z,
        labels=fake
      ))
      D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_x,
        labels=real
      ))

      D_loss /= 2.
    elif self.gan_strategy == 'lsgan':
      D_G_z = D_g
      D_x = D_x
      G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
      D_loss = tf.reduce_mean((D_x - 1.) ** 2)
      D_loss += tf.reduce_mean(D_G_z ** 2)
      D_loss /= 2.
    elif self.gan_strategy == 'wgangp':
      G_loss = -tf.reduce_mean(D_g)
      D_loss = tf.reduce_mean(D_g) - tf.reduce_mean(D_x)

      alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
      differences = D_E_x - x
      interpolates = x + (alpha * differences)
      
      with tf.name_scope('D_interp'), tf.variable_scope('Disc', reuse=True):
        D_interp = self.build_discriminator(interpolates)

      gradients = tf.gradients(D_interp, [interpolates])[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
      gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

      D_loss += self.wgangp_lambda * gradient_penalty
    else:
      raise ValueError()

    if self.gan_strategy == 'dcgan':
      G_opt = tf.train.AdamOptimizer(
          learning_rate=2e-4,
          beta1=0.5)
      D_opt = tf.train.AdamOptimizer(
          learning_rate=2e-4,
          beta1=0.5)
    elif self.gan_strategy == 'lsgan':
      G_opt = tf.train.RMSPropOptimizer(
          learning_rate=1e-4)
      D_opt = tf.train.RMSPropOptimizer(
          learning_rate=1e-4)
    elif self.gan_strategy == 'wgangp':
      G_opt = tf.train.AdamOptimizer(
          learning_rate=1e-4,
          beta1=0.5,
          beta2=0.9)
      D_opt = tf.train.AdamOptimizer(
          learning_rate=1e-4,
          beta1=0.5,
          beta2=0.9)
    else:
      raise NotImplementedError()

    self.l1 = l1 = tf.reduce_mean(tf.abs(x - input_mask * D_E_x))
    self.l2 = l2 = tf.reduce_mean(tf.square(x - input_mask * D_E_x))
    
    if self.objective == 'l1':
      recon_loss = l1
    elif self.objective == 'l2':
      recon_loss = l2

    self.step = step = tf.train.get_or_create_global_step()
    self.G_vars = G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
    self.D_vars = D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')

    G_loss_combined = G_loss + self.alpha * recon_loss
    self.G_train_op = G_opt.minimize(G_loss_combined, var_list=G_vars,
        global_step=tf.train.get_or_create_global_step())
    self.D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    # self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='AE') + [step]

    embedding_image = tf.image.rot90(tf.expand_dims(E_x[:, :, 0, :], -1))
    tf.summary.audio('clean', clean_audio[:, :, 0, :], self.audio_fs)
    tf.summary.audio('x', x[:, :, 0, :], self.audio_fs)
    tf.summary.audio('D_E_x', D_E_x[:, :, 0, :], self.audio_fs)
    tf.summary.image('E_x', embedding_image)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('l1', l1)
    tf.summary.scalar('l2', l2)
    tf.summary.scalar('loss', recon_loss)

  def train_loop(self, sess):
    sess.run(self.train_op)
