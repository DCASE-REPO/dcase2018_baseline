"""Input pipeline for DCASE 2018 Task 2 Baseline models."""

import functools
import os

import numpy as np
from scipy.io import wavfile
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as tf_audio

# All input clips use a 44.1 kHz sample rate.
SAMPLE_RATE = 44100

def clip_to_waveform(clip, clip_dir=None):
  """Decodes a WAV clip into a waveform tensor."""
  # Decode the WAV-format clip into a waveform tensor where
  # the values lie in [-1, +1].
  clip_path = tf.string_join([clip_dir, clip], separator=os.sep)
  clip_data = tf.read_file(clip_path)
  waveform, sr = tf_audio.decode_wav(clip_data)
  # Assert that the clip has the expected sample rate.
  check_sr = tf.assert_equal(sr, SAMPLE_RATE)
  # and that it is mono.
  check_channels = tf.assert_equal(tf.shape(waveform)[1], 1)
  with tf.control_dependencies([tf.group(check_sr, check_channels)]):
    return tf.squeeze(waveform)

def clip_to_log_mel_examples(clip, clip_dir=None, hparams=None):
  """Decodes a WAV clip into a batch of log mel spectrum examples."""
  # Decode WAV clip into waveform tensor.
  waveform = clip_to_waveform(clip, clip_dir=clip_dir)

  # Convert waveform into spectrogram using a Short-Time Fourier Transform.
  # Note that tf.contrib.signal.stft() uses a periodic Hann window by default.
  window_length_samples = int(round(SAMPLE_RATE * hparams.stft_window_seconds))
  hop_length_samples = int(round(SAMPLE_RATE * hparams.stft_hop_seconds))
  fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
  magnitude_spectrogram = tf.abs(tf.contrib.signal.stft(
      signals=waveform,
      frame_length=window_length_samples,
      frame_step=hop_length_samples,
      fft_length=fft_length))

  # Convert spectrogram into log mel spectrogram.
  num_spectrogram_bins = fft_length // 2 + 1
  linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
     num_mel_bins=hparams.mel_bands,
     num_spectrogram_bins=num_spectrogram_bins,
     sample_rate=SAMPLE_RATE,
     lower_edge_hertz=hparams.mel_min_hz,
     upper_edge_hertz=hparams.mel_max_hz)
  mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)
  log_mel_spectrogram = tf.log(mel_spectrogram + hparams.mel_log_offset)

  # Frame log mel spectrogram into examples.
  spectrogram_sr = 1 / hparams.stft_hop_seconds
  example_window_length_samples = int(round(spectrogram_sr * hparams.example_window_seconds))
  example_hop_length_samples = int(round(spectrogram_sr * hparams.example_hop_seconds))
  features = tf.contrib.signal.frame(
      signal=log_mel_spectrogram,
      frame_length=example_window_length_samples,
      frame_step=example_hop_length_samples,
      axis=0)

  return features

def record_to_labeled_log_mel_examples(csv_record, clip_dir=None, hparams=None,
                                       label_class_index_table=None, num_classes=None):
  """Creates a batch of log mel spectrum examples from a training record.

  Args:
    csv_record: a line from the train.csv file downloaded from Kaggle.
    clip_dir: path to a directory containing clips referenced by csv_record.
    hparams: tf.contrib.training.HParams object containing model hyperparameters.
    label_class_index_table: a lookup table that represents the class map.
    num_classes: number of classes in the class map.

  Returns:
    features: Tensor containing a batch of log mel spectrum examples.
    labels: Tensor containing corresponding labels in 1-hot format.
  """
  [clip, label, _] = tf.decode_csv(csv_record, record_defaults=[[''],[''],[0]])

  features = clip_to_log_mel_examples(clip, clip_dir=clip_dir, hparams=hparams)

  class_index = label_class_index_table.lookup(label)
  label_onehot = tf.one_hot(class_index, num_classes)
  num_examples = tf.shape(features)[0]
  labels = tf.tile([label_onehot], [num_examples, 1])

  return features, labels

def get_class_map(class_map_path):
  """Constructs a class label lookup table from a class map."""
  label_class_index_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.TextFileInitializer(
          filename=class_map_path,
          key_dtype=tf.string, key_index=1,
          value_dtype=tf.int32, value_index=0,
          delimiter=','),
      default_value=-1)
  num_classes = len(open(class_map_path).readlines())
  return label_class_index_table, num_classes

def train_input(train_csv_path=None, train_clip_dir=None, class_map_path=None, hparams=None):
  """Creates training input pipeline.

  Args:
    train_csv_path: path to the train.csv file provided by Kaggle.
    train_clip_dir: path to the unzipped audio_train/ directory from the
        audio_train.zip file provided by Kaggle.
    class_map_path: path to the class map prepared from the training data.
    hparams: tf.contrib.training.HParams object containing model hyperparameters

  Returns:
    features: Tensor containing a batch of log mel spectrum examples.
    labels: Tensor containing corresponding labels in 1-hot format.
    num_classes: number of classes.
    iter_init: an initializer op for the iterator that provides features and
       labels, to be run before the input pipeline is read.
  """
  label_class_index_table, num_classes = get_class_map(class_map_path)

  dataset = tf.data.TextLineDataset(train_csv_path)
  # Skip the header.
  dataset = dataset.skip(1)
  # Shuffle the list of clips. 10K is big enough to cover all clips.
  dataset = dataset.shuffle(buffer_size=10000)
  # Map each clip to a batch of framed log mel spectrum examples.
  dataset = dataset.map(
      map_func=functools.partial(
          record_to_labeled_log_mel_examples,
          clip_dir=train_clip_dir,
          hparams=hparams,
          label_class_index_table=label_class_index_table,
          num_classes=num_classes),
      # 4 is empirically chosen to use 4 logical CPU cores. Adjust as
      # needed if more or less resources are available.
      num_parallel_calls=4)
  # Unbatch so that we have a dataset of individual examples that we can then
  # shuffle for training. 20K should be enough to allow shuffling across a
  # few hundred clips which are already in random order.
  dataset = dataset.apply(tf.contrib.data.unbatch())
  dataset = dataset.shuffle(buffer_size=20000)
  # Run until we have completed 100 epochs of the training set.
  dataset = dataset.repeat(100)
  # Batch examples.
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=hparams.batch_size))
  # Let the input pipeline run a few batches ahead so that the model is
  # never starved of data.
  dataset = dataset.prefetch(10)

  iterator = dataset.make_initializable_iterator()
  features, labels = iterator.get_next()

  return features, labels, num_classes, iterator.initializer
