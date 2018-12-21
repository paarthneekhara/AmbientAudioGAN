"""
The idea behind this file is that it will become a *very messy* but *always reproducible* dumping ground for our feature extraction code.

Passing in a specific name to get_named_extractor should always result in the same extraction graph.

This extraction graph takes an audio tensor of shape [nsamps, 1, nch] and produces a feature tensor of shape [ntsteps, nfeats, nch]

Over time, the names for this will surely diverge into things like 'logmelspec_hop2000_icml2019_pleasework' but this is intended.
"""

import tensorflow as tf


def _stft(x, nfft, nhop):
  x = x[:, 0, :]
  x = tf.transpose(x, [1, 0])
  X = tf.contrib.signal.stft(x, nfft, nhop, pad_end=True)
  # X is [nch, ntsteps, nfeats]
  X = tf.transpose(X, [1, 2, 0])
  return X

  
def get_named_extractor(name, audio_fs):
  if name == 'linspec':
    nfft = 256
    nhop = 160
    def feature_extractor(x):
      X = _stft(x, nfft, nhop)
      X_mag = tf.abs(X)
      return (x, X_mag)
    feature_fs = float(audio_fs) / nhop
    return feature_fs, feature_extractor

  elif name == 'logspec':
    nfft = 256
    nhop = 160
    def feature_extractor(x):
      X = _stft(x, nfft, nhop)
      X_mag = tf.abs(X)
      X_lmag = tf.log(X_mag + 1e-8)
      return (x, X_lmag)
    feature_fs = float(audio_fs) / nhop
    return feature_fs, feature_extractor

  elif name == 'melspec':
    raise NotImplementedErorr()

  elif name == 'logmelspec':
    raise NotImplementedErorr()

  else:
    raise NotImplementedErorr()
