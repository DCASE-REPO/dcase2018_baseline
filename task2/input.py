
import functools
import os

import numpy as np
from scipy.io import wavfile
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as tf_audio

SAMPLE_RATE = 44100

def clip_to_waveform(clip, clip_dir=None):
  clip_path = tf.string_join([clip_dir, clip], separator=os.sep)
  clip_data = tf.read_file(clip_path)
  waveform, sr = tf_audio.decode_wav(clip_data)
  check_sr = tf.assert_equal(sr, SAMPLE_RATE)
  check_channels = tf.assert_equal(tf.shape(waveform)[1], 1)
  with tf.control_dependencies([tf.group(check_sr, check_channels)]):
    return tf.squeeze(waveform)

def clip_to_log_mel_examples(clip, clip_dir=None, hparams=None):
  waveform = clip_to_waveform(clip, clip_dir=clip_dir)

  window_length_samples = int(round(SAMPLE_RATE * hparams.stft_window_seconds))
  hop_length_samples = int(round(SAMPLE_RATE * hparams.stft_hop_seconds))
  fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
  magnitude_spectrogram = tf.abs(tf.contrib.signal.stft(
      signals=waveform,
      frame_length=window_length_samples,
      frame_step=hop_length_samples,
      fft_length=fft_length))

  num_spectrogram_bins = fft_length // 2 + 1
  linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
     num_mel_bins=hparams.mel_bands,
     num_spectrogram_bins=num_spectrogram_bins,
     sample_rate=SAMPLE_RATE,
     lower_edge_hertz=hparams.mel_min_hz,
     upper_edge_hertz=hparams.mel_max_hz)
  mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)
  log_mel_spectrogram = tf.log(mel_spectrogram + hparams.mel_log_offset)

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
  [clip, label, _] = tf.decode_csv(csv_record, record_defaults=[[''],[''],[0]])

  features = clip_to_log_mel_examples(clip, clip_dir=clip_dir, hparams=hparams)

  class_index = label_class_index_table.lookup(label)
  label_onehot = tf.one_hot(class_index, num_classes)
  num_examples = tf.shape(features)[0]
  labels = tf.tile([label_onehot], [num_examples, 1])

  return features, labels

def get_class_map(class_map_path):
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
  label_class_index_table, num_classes = get_class_map(class_map_path)

  dataset = tf.data.TextLineDataset(train_csv_path)
  dataset = dataset.skip(1)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.map(
      map_func=functools.partial(
          record_to_labeled_log_mel_examples,
          clip_dir=train_clip_dir,
          hparams=hparams,
          label_class_index_table=label_class_index_table,
          num_classes=num_classes),
      num_parallel_calls=1)
  dataset = dataset.apply(tf.contrib.data.unbatch())
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.repeat(100)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=hparams.batch_size))
  dataset = dataset.prefetch(10)

  iterator = dataset.make_initializable_iterator()
  features, labels = iterator.get_next()

  return features, labels, num_classes, iterator.initializer


if __name__ == '__main__':
  dataset_root = '/usr/local/google/home/plakal/fsd12k/final/dataset'
  train_csv_path = os.path.join(dataset_root, 'dev', 'dataset_train.csv')
  train_clip_dir = os.path.join(dataset_root, 'dev', 'audio')
  class_map_path = '/usr/local/google/home/plakal/fsd12k/baseline/class_map.csv'
  hparams = tf.contrib.training.HParams(
      stft_window_seconds=0.025,
      stft_hop_seconds=0.010,
      mel_bands=64,
      mel_min_hz=125,
      mel_max_hz=7500,
      mel_log_offset=0.001,
      example_window_seconds=0.250,
      example_hop_seconds=0.125,
      batch_size=16)

  with tf.Graph().as_default(), tf.Session() as sess:
    features, labels, num_classes, input_init = train_input(
        train_csv_path=train_csv_path, train_clip_dir=train_clip_dir, class_map_path=class_map_path,
        hparams=hparams)
    sess.run(tf.tables_initializer())
    sess.run(input_init)
    while True:
      try:
        features_data, labels_data = sess.run([features, labels])
        print("features", features_data.shape)
        print("labels", labels_data.shape)
        print(features_data)
        print(labels_data)
      except tf.errors.OutOfRangeError:
        print('read all files')
        break
