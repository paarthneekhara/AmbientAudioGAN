import time

import tensorflow as tf

from wgpp.loader import waveform_decoder
from wgpp.model import Modes
from wgpp.util import override_model_attrs
from waveAE import WaveAE


def infer(args):
  infer_dir = os.path.join(args.train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Initialize model
  model = WaveAE(Modes.INFER)
  model, summary = override_model_attrs(model, args.model_overrides)

  x = tf.placeholder(tf.float32, [None, model.subseq_len, 1, 1], name='x')

  # Create model
  model(x)

  # Name important tensors
  E_x = tf.identity(model.E_x, name='E_x')
  E_x_img = tf.transpose(E_x, [0, 1, 3, 2])
  E_x_img = tf.image.rot90(E_x_img)
  E_x_img += 1
  E_x_img /= 2.
  E_x_img *= 255.
  E_x_img = tf.clip_by_value(E_x_img, 0., 255.)
  E_x_img = tf.cast(E_x_img, tf.uint8, name='E_x_img')
  D_E_x = tf.identity(model.D_E_x, name='D_E_x')

  # Create saver
  saver = tf.train.Saver(var_list=model.all_vars)

  # Export graph
  tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

  # Export MetaGraph
  infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  tf.train.export_meta_graph(
      filename=infer_metagraph_fp,
      clear_devices=True,
      saver_def=saver.as_saver_def())

  # Reset graph (in case training afterwards)
  tf.reset_default_graph()


def train(fps, args):
  # Initialize model
  model = WaveAE(Modes.TRAIN)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  # Load data
  with tf.name_scope('loader'):
    x = waveform_decoder(
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
  model(x)

  # Train
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=60,
      save_summaries_secs=5) as sess:
    while not sess.should_stop():
      model.train_loop(sess)


def eval(fps, args):
  if args.eval_dataset_name is not None:
    eval_dir = os.path.join(args.train_dir,
        'eval_{}'.format(args.eval_dataset_name))
  else:
    eval_dir = os.path.join(args.train_dir, 'eval_valid')
  if not os.path.isdir(eval_dir):
    os.makedirs(eval_dir)

  # Initialize model
  model = WaveAE(Modes.EVAL)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  # Load data
  with tf.name_scope('loader'):
    x = waveform_decoder(
        fps=fps,
        batch_size=model.eval_batch_size,
        subseq_len=model.subseq_len,
        audio_fs=model.audio_fs,
        audio_mono=True,
        audio_normalize=True,
        decode_fastwav=args.data_fastwav,
        decode_parallel_calls=1,
        repeat=False,
        shuffle=False,
        shuffle_buffer_size=None,
        subseq_randomize_offset=False,
        subseq_overlap_ratio=0.,
        subseq_pad_end=True,
        prefetch_size=None,
        gpu_num=None)

  # Create model
  model(x)

  # Create saver and summary writer
  saver = tf.train.Saver(var_list=model.all_vars, max_to_keep=1)
  summary_writer = tf.summary.FileWriter(eval_dir)

  ckpt_fp = None
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      ckpt_fp = latest_ckpt_fp
      print('Evaluating {}'.format(ckpt_fp))
      with tf.Session() as sess:
        model.eval_ckpt(ckpt_fp, sess, summary_writer, saver, eval_dir)
      print('Done!')
    time.sleep(1)
  

if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import os

  parser = ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'eval', 'infer'])
  parser.add_argument('train_dir', type=str)

  parser.add_argument('--data_dir', type=str, required=True)
  parser.add_argument('--data_fastwav', dest='data_fastwav', action='store_true')
  parser.add_argument('--data_randomize_offset', dest='data_randomize_offset', action='store_true')
  parser.add_argument('--data_overlap_ratio', type=float)

  parser.add_argument('--model_overrides', type=str)

  parser.add_argument('--train_ckpt_every_nsecs', type=int)
  parser.add_argument('--train_summary_every_nsecs', type=int)

  parser.add_argument('--eval_dataset_name', type=str)

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
      eval_dataset_name=None)

  args = parser.parse_args()

  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  fps = glob.glob(os.path.join(args.data_dir, '*.wav'))
  print('Found {} audio files'.format(len(fps)))

  if args.mode == 'train':
    infer(args)
    train(fps, args)
  elif args.mode == 'eval':
    eval(fps, args)
  elif args.mode == 'infer':
    infer(args)
