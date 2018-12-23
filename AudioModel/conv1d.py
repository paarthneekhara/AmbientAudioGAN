import tensorflow as tf
import math

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
      enc_length = 64,
      nonlin=tf.nn.tanh):
    self.dim = dim
    self.kernel_len = kernel_len
    self.stride = stride
    self.batchnorm = batchnorm
    self.nonlin = nonlin
    self.enc_length = enc_length

  def __call__(self, x, training=False):
    conv1d = lambda x, n: conv1d_layer(x, n, self.kernel_len, self.stride)
    conv1x1d = lambda x, n: conv1d_layer(x, n, 1, 1)

    if self.batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, training=training)
    else:
      batchnorm = lambda x: x

    print("Encoder")
    
    n_layers = int((math.log(16384./self.enc_length)/math.log(self.stride)))
    print (n_layers)
    
    encoder_activations = [] # (4096, 1024,..,64)
    for ln in range(n_layers):
      with tf.variable_scope('downconv_{}'.format(ln)):
        x = conv1d(x, self.dim * (2**ln))
        encoder_activations.append(x)
      x = tf.nn.leaky_relu(x)
      x = batchnorm(x)
      print(x)

    self.encoder_activations = encoder_activations
    print ("Encoder Activations")
    for enc_ac in encoder_activations:
      print (enc_ac)
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
      use_skip = False,
      encoder_activations = [],
      enc_length = 64,
      nonlin=tf.nn.tanh):
    self.dim = dim
    self.kernel_len = kernel_len
    self.stride = stride
    self.batchnorm = batchnorm
    self.nonlin = nonlin
    self.use_skip = use_skip
    self.encoder_activations = encoder_activations
    self.enc_length = enc_length
    
  def __call__(self, enc_x, training=False):
    conv1d_transpose = lambda x, n: conv1d_transpose_layer(x, n, self.kernel_len, self.stride)
    conv1x1d_transpose = lambda x, n: conv1d_transpose_layer(x, n, 1, 1)

    if self.batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, training=training)
    else:
      batchnorm = lambda x: x

    print("Decoder")
    x = enc_x
    print(x)
    # Deaggregation layer
    # e.g. [64, 64] -> [64, 512]
    n_layers = int((math.log(16384./self.enc_length)/math.log(self.stride)))
    channels_initial = int(self.dim * (2**(n_layers-1)))

    encoder_activations = self.encoder_activations

    channels = channels_initial    
    with tf.variable_scope('upconv_1x1'):
      x = conv1x1d_transpose(x, channels)
      if self.use_skip:
        print("Adding Skip Connection {}".format(0))
        print("Encoder Act", encoder_activations[-1])
        print("Decoder Act", x)
        x += encoder_activations[-1]

    x = tf.nn.relu(x)
    x = batchnorm(x)
    print(x)
    # Layer 1
    # [64, 512] -> [256, 256]
    

    # channels = 
    for ln in range(n_layers - 1):
      channels = int(channels/2.)
      with tf.variable_scope('upconv_{}'.format(ln)):
        x = conv1d_transpose(x, channels)
        if self.use_skip:
          print("Adding Skip Connection {}".format(ln))
          print("Encoder Act", encoder_activations[n_layers-ln-2])
          print("Decoder Act", x)
          x += encoder_activations[n_layers-ln-2]

      x = tf.nn.relu(x)
      x = batchnorm(x)
      print(x)

    with tf.variable_scope('upconv_{}'.format(n_layers - 1)):
      x = conv1d_transpose(x, 1)


    if self.nonlin is not None:
      x = self.nonlin(x)

    print(x)
    return x

