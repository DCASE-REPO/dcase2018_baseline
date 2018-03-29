#!/usr/bin/env python

import csv
from collections import defaultdict
import os
import sys

import numpy as np
import tensorflow as tf

import input
import model

TOP_N = 3

def inference(model_name=None, hparams=None, test_clip_dir=None,
              class_map_path=None, checkpoint_path=None, predictions_csv_path=None):
  with tf.Graph().as_default():
    _, num_classes = input.get_class_map(class_map_path)
    clip = tf.placeholder(tf.string, [])
    features = input.clip_to_log_mel_examples(
        clip, clip_dir=test_clip_dir, hparams=hparams)
    _, prediction, _, _ = model.define_model(
        model_name=model_name, features=features, num_classes=num_classes,
        hparams=hparams, training=False)

    saver = tf.train.Saver()

    with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=checkpoint_path) as sess:
      class_map = {int(row[0]): row[1] for row in csv.reader(open(class_map_path))}
      test_clips = sorted(os.listdir(test_clip_dir))
      pred_writer = csv.DictWriter(open(predictions_csv_path, 'w'), fieldnames=['fname', 'label'])
      for (i, test_clip) in enumerate(test_clips):
        print(i, test_clip)
        sys.stdout.flush()
        # Hack to avoid passing empty files through the model.
        if os.path.getsize(os.path.join(test_clip_dir, test_clip)) == 44:
          print('empty file, skipped model')
          label = ''
        else:
          predicted = sess.run(prediction, {clip: test_clip})
          predicted = np.average(predicted, axis=0)
          predicted_classes = np.argsort(predicted)[::-1][:TOP_N]
          label = ' '.join([class_map[c] for c in predicted_classes])
          pred_writer.writerow({'fname': test_clip, 'label': label})
        print(label)
        sys.stdout.flush()


if __name__ == '__main__':
  test_clip_dir = '/usr/local/google/home/plakal/fsd12k/kaggledata/audio_test'
  class_map_path = '/usr/local/google/home/plakal/fsd12k/train/class_map.csv'
  model_name = 'cnn'
  hparams = tf.contrib.training.HParams(
      stft_window_seconds=0.025,
      stft_hop_seconds=0.010,
      mel_bands=64,
      mel_min_hz=125,
      mel_max_hz=7500,
      mel_log_offset=0.001,
      example_window_seconds=0.250,
      example_hop_seconds=0.125,
      batch_size=64,
      nl=2,
      nh=256,
      lr=1e-5,
      adam_eps=1e-8)
  checkpoint_path = sys.argv[1]  # '/usr/local/google/home/plakal/fsd12k/train/model.ckpt-1001'
  predictions_csv_path = '/usr/local/google/home/plakal/fsd12k/train/predictions.csv'

  inference(model_name=model_name, hparams=hparams,
            test_clip_dir=test_clip_dir, class_map_path=class_map_path,
            checkpoint_path=checkpoint_path, predictions_csv_path=predictions_csv_path)
