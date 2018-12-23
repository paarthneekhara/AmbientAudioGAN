import time
import pickle

import numpy as np
import tensorflow as tf

from AudioModel.loader_measured import waveform_decoder
from AudioModel.model import Modes
from AudioModel.util import override_model_attrs
from EncDecModel import WaveAE

def train(fps, args):
  # Initialize model
  model = WaveAE(Modes.TRAIN)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  # Load data
  with tf.name_scope('loader'):
    clean, x = waveform_decoder(
        fps=fps,
        batch_size=model.train_batch_size,
        subseq_len=model.subseq_len,
        audio_fs=model.audio_fs,
        audio_mono=True,
        audio_normalize=True,
        decode_fastwav=args.data_fastwav,
        decode_parallel_calls=4,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=4096,
        subseq_randomize_offset=args.data_randomize_offset,
        subseq_overlap_ratio=args.data_overlap_ratio,
        subseq_pad_end=True,
        prefetch_size=64,
        gpu_num=0)

  # Create model
  model(clean, x)

  # Train
  # model_dir_path = "/data2/paarth/TrainDir/WaveAE/WaveAEsc09_l1batchnormFalse/eval_sc09_valid"
  # ckpt = 253802
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_ckpt_every_nsecs,
      save_summaries_secs=args.train_summary_every_nsecs) as sess:
    while not sess.should_stop():
      model.train_loop(sess)
  

if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import os

  parser = ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'incept', 'infer'])
  parser.add_argument('train_dir', type=str)

  parser.add_argument('--data_dir', type=str, required=True)
  parser.add_argument('--data_fastwav', dest='data_fastwav', action='store_true')
  parser.add_argument('--data_randomize_offset', dest='data_randomize_offset', action='store_true')
  parser.add_argument('--data_overlap_ratio', type=float)

  parser.add_argument('--model_overrides', type=str)

  parser.add_argument('--train_ckpt_every_nsecs', type=int)
  parser.add_argument('--train_summary_every_nsecs', type=int)


  parser.set_defaults(
      mode=None,
      train_dir=None,
      data_dir=None,
      data_fastwav=False,
      data_randomize_offset=False,
      data_overlap_ratio=0.,
      model_cfg_overrides=None,
      train_ckpt_every_nsecs=360,
      train_summary_every_nsecs=60,
      )

  args = parser.parse_args()

  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  fps = glob.glob(os.path.join(args.data_dir, '*.wav'))
  print('Found {} audio files'.format(len(fps)))

  if args.mode == 'train':
    train(fps, args)
  else:
    raise NotImplementedError()
