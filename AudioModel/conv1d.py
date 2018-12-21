import tensorflow as tf


def conv1d_layer(
    x,
    num_output_ch,
    kernel_len,
    stride,
    padding='same',
    **kwargs):
  return tf.layers.conv2d(
      x,
      num_output_ch,
      (kernel_len, 1),
      strides=(stride, 1),
      padding=padding,
      **kwargs)


def conv1d_transpose_layer(
    x,
    num_output_ch,
    kernel_len,
    stride,
    padding='same',
    **kwargs):
  return tf.layers.conv2d_transpose(
      x,
      num_output_ch,
      (kernel_len, 1),
      strides=(stride, 1),
      padding=padding,
      **kwargs)


class WaveEncoderFactor256(object):
  def __init__(
      self,
      dim=64,
      kernel_len=25,
      stride=4,
      batchnorm=False,
      nonlin=tf.nn.tanh):
    self.dim = dim
    self.kernel_len = kernel_len
    self.stride = stride
    self.batchnorm = batchnorm
    self.nonlin = nonlin


  def __call__(self, x, training=False):
    conv1d = lambda x, n: conv1d_layer(x, n, self.kernel_len, self.stride)
    conv1x1d = lambda x, n: conv1d_layer(x, n, 1, 1)

    if self.batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, training=training)
    else:
      batchnorm = lambda x: x

    # Layer 0
    # e.g. [16384, 1] -> [4096, 64]
    with tf.variable_scope('downconv_0'):
      x = conv1d(x, self.dim)
    x = tf.nn.leaky_relu(x)
    x = batchnorm(x)

    # Layer 1
    # [4096, 64] -> [1024, 128]
    with tf.variable_scope('downconv_1'):
      x = conv1d(x, self.dim * 2)
    x = tf.nn.leaky_relu(x)
    x = batchnorm(x)

    # Layer 2
    # [1024, 128] -> [256, 256]
    with tf.variable_scope('downconv_2'):
      x = conv1d(x, self.dim * 4)
    x = tf.nn.leaky_relu(x)
    x = batchnorm(x)

    # Layer 3
    # [256, 256] -> [64, 512]
    with tf.variable_scope('downconv_3'):
      x = conv1d(x, self.dim * 8)
    x = tf.nn.leaky_relu(x)
    x = batchnorm(x)

    # Aggregation layer
    # [64, 512] -> [64, 64]
    with tf.variable_scope('downconv_1x1'):
      x = conv1x1d(x, self.dim * 1)

    if self.nonlin is not None:
      x = self.nonlin(x)

    return x


class WaveDecoderFactor256(object):
  def __init__(
      self,
      dim=64,
      kernel_len=25,
      stride=4,
      batchnorm=False,
      nonlin=tf.nn.tanh):
    self.dim = dim
    self.kernel_len = kernel_len
    self.stride = stride
    self.batchnorm = batchnorm
    self.nonlin = nonlin


  def __call__(self, enc_x, training=False):
    conv1d_transpose = lambda x, n: conv1d_transpose_layer(x, n, self.kernel_len, self.stride)
    conv1x1d_transpose = lambda x, n: conv1d_transpose_layer(x, n, 1, 1)

    if self.batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, training=training)
    else:
      batchnorm = lambda x: x

    x = enc_x
    # Deaggregation layer
    # e.g. [64, 64] -> [64, 1024]
    with tf.variable_scope('upconv_1x1'):
      x = conv1x1d_transpose(x, self.dim * 8)
    x = tf.nn.relu(x)
    x = batchnorm(x)

    # Layer 1
    # [64, 512] -> [256, 256]
    with tf.variable_scope('upconv_1'):
      x = conv1d_transpose(x, self.dim * 4)
    x = tf.nn.relu(x)
    x = batchnorm(x)

    # Layer 2
    # [256, 256] -> [1024, 128]
    with tf.variable_scope('upconv_2'):
      x = conv1d_transpose(x, self.dim * 2)
    x = tf.nn.relu(x)
    x = batchnorm(x)

    # Layer 3
    # [1024, 128] -> [4096, 64]
    with tf.variable_scope('upconv_3'):
      x = conv1d_transpose(x, self.dim)
    x = tf.nn.relu(x)
    x = batchnorm(x)

    # Layer 4
    # [4096, 64] -> [16384, 1]
    with tf.variable_scope('upconv_4'):
      x = conv1d_transpose(x, 1)

    if self.nonlin is not None:
      x = self.nonlin(x)

    return x

