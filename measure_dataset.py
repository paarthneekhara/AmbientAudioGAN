import time
import pickle
import numpy as np
import tensorflow as tf
from AudioModel.loader import waveform_decoder
import measurement
import scipy.io.wavfile

def measure(x, args):
  signal = x[:,:,0,:]
  if args.measurement == 'block_patch':
    measured_audio, _ = measurement.block_patch(
      signal, 
      patch_size = args.m_patch_size)
  elif args.measurement == 'drop_patches':
    measured_audio, _ = measurement.drop_patches(
      signal, 
      patch_size = args.m_patch_size, 
      drop_prob = args.m_prob)
  else:
    raise NotImplementedError() 

  return signal, measured_audio

def main(fps, args):
  with tf.name_scope('loader'):
    x = waveform_decoder(
        fps=fps,
        batch_size=1,
        subseq_len=args.subseq_len,
        audio_fs=args.audio_fs,
        audio_mono=True,
        audio_normalize=True,
        decode_fastwav=args.data_fastwav,
        decode_parallel_calls=4,
        repeat=False,
        shuffle=False,
        shuffle_buffer_size=None,
        subseq_randomize_offset=False,
        subseq_overlap_ratio=args.data_overlap_ratio,
        subseq_pad_end=True,
        prefetch_size=64,
        gpu_num=0)

  # Create model
  t_clean, t_measured = measure(x, args)
  with tf.Session() as sess:
    print("session created")
    file_no = 0
    while True:
        try:
          clean, measured = sess.run([t_clean, t_measured])
          file_no += 1
          scipy.io.wavfile.write(os.path.join(args.dest_dir, "{}_clean.wav".format(file_no)), args.audio_fs, clean[0,:,0])
          scipy.io.wavfile.write(os.path.join(args.dest_dir, "{}_measured.wav".format(file_no)), args.audio_fs, measured[0,:,0])
          print (file_no)
        except tf.errors.OutOfRangeError:
          break
    print("Done")
  

if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import os

  parser = ArgumentParser()

  
  parser.add_argument('data_dir', type=str)
  parser.add_argument('dest_dir', type=str)

  # parser.add_argument('--data_dir', type=str, required=True)
  parser.add_argument('--data_fastwav', dest='data_fastwav', action='store_true')
  parser.add_argument('--data_randomize_offset', dest='data_randomize_offset', action='store_true')
  parser.add_argument('--data_overlap_ratio', type=float)

  parser.add_argument('--model_overrides', type=str)

  parser.add_argument('--train_ckpt_every_nsecs', type=int)
  parser.add_argument('--train_summary_every_nsecs', type=int)

  # parser.add_argument('--batch_size', type=int)
  parser.add_argument('--subseq_len', type=int)
  parser.add_argument('--audio_fs', type=int)

  parser.add_argument('--measurement', type=str)
  parser.add_argument('--m_patch_size', type=int)
  parser.add_argument('--m_prob', type=float)


  parser.set_defaults(
      data_dir=None,
      dest_dir=None,
      data_fastwav=False,
      data_randomize_offset=False,
      data_overlap_ratio=0.,
      model_cfg_overrides=None,
      train_ckpt_every_nsecs=360,
      train_summary_every_nsecs=60,
      # batch_size = 1,
      subseq_len = 16384,
      audio_fs = 16000,
      measurement = 'drop_patches',
      m_patch_size = 512,
      m_prob = 0.5
      )

  args = parser.parse_args()

  if not os.path.isdir(args.dest_dir):
    os.makedirs(args.dest_dir)

  fps = glob.glob(os.path.join(args.data_dir, '*.wav'))
  print('Found {} audio files'.format(len(fps)))

  
  main(fps, args)
  
