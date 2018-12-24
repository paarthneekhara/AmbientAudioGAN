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


class WaveEncoder(object):
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
        if self.stride == 4:
          x = conv1d(x, self.dim * (2**ln))
        elif self.stride == 2:
          lne = int((ln)/2)
          x = conv1d(x, self.dim * (2**lne))
        encoder_activations.append(x)
      
      if ln != n_layers - 1:
        x = tf.nn.leaky_relu(x)
        x = batchnorm(x)
      print(x)

    self.encoder_activations = encoder_activations
    if self.nonlin is not None:
      x = self.nonlin(x)

    return x


class WaveDecoder(object):
  def __init__(
      self,
      dim=64,
      kernel_len=25,
      stride=4,
      batchnorm=False,
      use_skip = False,
      encoder_activations = [],
      enc_length = 64,
      skip_limit = 3,
      nonlin=tf.nn.tanh):
    self.dim = dim
    self.kernel_len = kernel_len
    self.stride = stride
    self.batchnorm = batchnorm
    self.nonlin = nonlin
    self.use_skip = use_skip
    self.encoder_activations = encoder_activations
    self.enc_length = enc_length
    self.skip_limit = skip_limit

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
    if self.stride == 4:
      channels_initial = int(self.dim * (2**(n_layers-1)))
    elif self.stride == 2:
      channels_initial = int(self.dim * (2**  int((n_layers-1)/2) ))

    encoder_activations = self.encoder_activations

    
    # channels = 
    n_skips = 0
    for ln in range(n_layers - 1):
      enc_ln = n_layers - 2 - ln
      if self.stride == 2:
        channels = int(self.dim * (2**int(enc_ln/2) ))
      else:
        channels = int(self.dim * (2**enc_ln)) 
      # channels = int(channels/2.)
      with tf.variable_scope('upconv_{}'.format(ln)):
        x = conv1d_transpose(x, channels)
        if self.use_skip and n_skips < self.skip_limit:
          print("Adding Skip Connection {}".format(ln))
          print("Encoder Act", encoder_activations[n_layers-ln-2])
          print("Decoder Act", x)
          x += encoder_activations[n_layers-ln-2]
          n_skips += 1
      x = tf.nn.relu(x)
      x = batchnorm(x)
      print(x)

    with tf.variable_scope('upconv_{}'.format(n_layers - 1)):
      x = conv1d_transpose(x, 1)


    if self.nonlin is not None:
      x = self.nonlin(x)

    print(x)
    return x

