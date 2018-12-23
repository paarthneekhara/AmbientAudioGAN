from scipy.io.wavfile import read as wavread
import numpy as np

import tensorflow as tf


def decode_audio(fp, fs=None, mono=False, normalize=False, fastwav=False, measured = False):
  """Decodes audio file paths into 32-bit floating point vectors.

  Args:
    fp: Audio file path.
    fs: If specified, resamples decoded audio to this rate.
    mono: If true, averages channels to mono.
    fastwav: Assume fp is a standard WAV file (PCM 16-bit or float 32-bit).

  Returns:
    A np.float32 array containing the audio samples at specified sample rate.
  """
  if measured:
    fp = fp.decode('latin').replace("clean", "measured")

  if fastwav:
    # Read with scipy wavread (fast).
    _fs, _wav = wavread(fp)
    if fs is not None and fs != _fs:
      raise NotImplementedError('Fastwav cannot resample audio.')
    if _wav.dtype == np.int16:
      _wav = _wav.astype(np.float32)
      _wav /= 32768.
    elif _wav.dtype == np.float32:
      pass
    else:
      raise NotImplementedError('Fastwav cannot process atypical WAV files.')
  else:
    # TODO: librosa currently optional due to issue with cluster installation
    import librosa
    # Decode with librosa load (slow but supports file formats like mp3).
    _wav, _fs = librosa.core.load(fp, sr=fs, mono=False)
    if _wav.ndim == 2:
      _wav = np.swapaxes(_wav, 0, 1)

  assert _wav.dtype == np.float32

  # At this point, _wav is np.float32 either [nsamps,] or [nsamps, nch].
  # We want [nsamps, 1, nch] to mimic 2D shape of spectral feats.
  if _wav.ndim == 1:
    nsamps = _wav.shape[0]
    nch = 1
  else:
    nsamps, nch = _wav.shape
  _wav = np.reshape(_wav, [nsamps, 1, nch])
 
  # Average channels if we want monaural audio.
  if mono:
    _wav = np.mean(_wav, 2, keepdims=True)

  if normalize:
    _wav /= np.max(np.abs(_wav))

  return _wav


def decode_extract_and_batch(
    fps,
    batch_size,
    subseq_len,
    audio_fs,
    audio_mono=True,
    audio_normalize=True,
    decode_fastwav=False,
    decode_parallel_calls=1,
    repeat=False,
    shuffle=False,
    shuffle_buffer_size=None,
    subseq_randomize_offset=False,
    subseq_overlap_ratio=0,
    subseq_pad_end=False,
    prefetch_size=None,
    gpu_num=None):
  """Decodes audio file paths into mini-batches of samples.

  Args:
    fps: List of audio file paths.
    batch_size: Number of items in the batch.
    subseq_len: Length of the subsequences in samples or feature timesteps.
    audio_fs: Sample rate for decoded audio files.
    audio_mono: If false, preserves multichannel (all files must have same).
    audio_normalize: If false, do not normalize audio waveforms.
    decode_fastwav: If true, uses scipy to decode standard wav files.
    decode_parallel_calls: Number of parallel decoding threads.
    repeat: If true (for training), continuously iterate through the dataset.
    shuffle: If true (for training), buffer and shuffle the subsequences.
    subseq_randomize_offset: If true, randomize starting position for subseq.
    pad_end: If true, allows zero-padded examples at the end.

  Returns:
    A tuple of np.float32 tensors representing audio and feature subsequences.
      audio: [batch_size, ?, 1, nch]
  """
  # Create dataset of filepaths
  dataset = tf.data.Dataset.from_tensor_slices(fps)

  # Shuffle all filepaths every epoch
  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(fps))

  # Repeat
  if repeat:
    dataset = dataset.repeat()

  def _decode_audio_shaped(fp):
    _decode_audio_clean = lambda _fp: decode_audio(
      _fp,
      fs=audio_fs,
      mono=audio_mono,
      normalize=audio_normalize,
      fastwav=decode_fastwav)

    _decode_audio_measured = lambda _fp: decode_audio(
      _fp,
      fs=audio_fs,
      mono=audio_mono,
      normalize=audio_normalize,
      fastwav=decode_fastwav,
      measured = True
      )

    clean_audio = tf.py_func(
        _decode_audio_clean,
        [fp],
        tf.float32,
        stateful=False)

    # m_fp = fp.replace("clean", "measured")
    measured_audio = tf.py_func(
        _decode_audio_measured,
        [fp],
        tf.float32,
        stateful=False)

    clean_audio.set_shape([None, 1, 1 if audio_mono else None])
    measured_audio.set_shape([None, 1, 1 if audio_mono else None])

    return clean_audio, measured_audio

  # Decode audio
  dataset = dataset.map(
      _decode_audio_shaped,
      num_parallel_calls=decode_parallel_calls)

  # Parallel
  def _subseq(clean_audio, measured_audio):
    # Calculate hop size
    assert subseq_overlap_ratio >= 0
    subseq_hop = int(round(subseq_len * (1. - subseq_overlap_ratio)))
    if subseq_hop < 1:
      raise ValueError('Overlap ratio too high')

    # Randomize starting phase:
    if subseq_randomize_offset:
      start = tf.random_uniform([], maxval=subseq_len, dtype=tf.int32)
      clean_audio = clean_audio[start:]
      measured_audio = measured_audio[start:]

    # Extract subsequences
    clean_audio_subseqs = tf.contrib.signal.frame(
        clean_audio,
        subseq_len,
        subseq_hop,
        pad_end=subseq_pad_end,
        pad_value=0,
        axis=0)
    measured_audio_subseqs = tf.contrib.signal.frame(
        measured_audio,
        subseq_len,
        subseq_hop,
        pad_end=subseq_pad_end,
        pad_value=0,
        axis=0)

    return clean_audio_subseqs, measured_audio_subseqs

  def _subseq_dataset_wrapper(clean_audio, measured_audio):
    clean_audio_subseqs, measured_audio_subseqs = _subseq(clean_audio, measured_audio)
    return tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(clean_audio_subseqs),
      tf.data.Dataset.from_tensor_slices(measured_audio_subseqs),
    ))
    

  # Extract parallel subsequences from both audio and features
  dataset = dataset.flat_map(_subseq_dataset_wrapper)

  # Shuffle examples
  if shuffle:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

  # Make batches
  dataset = dataset.batch(batch_size, drop_remainder=True)

  # Queue up a number of batches on the CPU side
  if prefetch_size is not None:
    dataset = dataset.prefetch(prefetch_size)
    if (gpu_num is not None) and tf.test.is_gpu_available():
      dataset = dataset.apply(
          tf.data.experimental.prefetch_to_device(
            '/device:GPU:{}'.format(gpu_num)))

  # Get tensors
  iterator = dataset.make_one_shot_iterator()
  # return dataset
  return iterator.get_next()


def waveform_decoder(*args, **kwargs):
  return decode_extract_and_batch(*args, **kwargs)
