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

def avg_precision(actual=None, predicted=None):
  for (i, p) in enumerate(predicted):
    if actual == p:
      return 1.0 / (i + 1.0)
  return 0.0

def print_maps(ap_sums=None, ap_counts=None, class_map=None):
  map_count = 0
  map_sum = 0.0
  for class_index in sorted(ap_counts.keys()):
    m_ap = ap_sums[class_index] / ap_counts[class_index]
    print("MAP for %s: %.4f" % (class_map[class_index], m_ap))
    map_count += ap_counts[class_index]
    map_sum += ap_sums[class_index]
  m_ap = map_sum / map_count
  print("Overall MAP: %.4f" % m_ap)

def eval(model_name=None, hparams=None, test_csv_path=None, test_clip_dir=None,
         class_map_path=None, checkpoint_path=None):
  with tf.Graph().as_default():
    label_class_index_table, num_classes = input.get_class_map(class_map_path)
    csv_record = tf.placeholder(tf.string, [])
    features, labels = input.record_to_labeled_log_mel_examples(
        csv_record, clip_dir=test_clip_dir, hparams=hparams,
        label_class_index_table=label_class_index_table, num_classes=num_classes)
    global_step, prediction, _, _ = model.define_model(
        model_name=model_name, features=features, num_classes=num_classes,
        hparams=hparams, training=False)

    saver = tf.train.Saver()

    with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=checkpoint_path) as sess:
      class_map = {int(row[0]): row[1] for row in csv.reader(open(class_map_path))}
      ap_counts = defaultdict(int)
      ap_sums = defaultdict(float)
      test_records = open(test_csv_path).readlines()
      for (i,record) in enumerate(test_records[1:]):
        record = record.strip()
        actual, predicted = sess.run([labels, prediction], {csv_record: record})

        actual_class = np.argmax(actual[0])
        predicted = np.average(predicted, axis=0)
        predicted_classes = np.argsort(predicted)[::-1][:TOP_N]
        ap = avg_precision(actual=actual_class, predicted=predicted_classes)
        print(actual_class, predicted_classes, ap)

        ap_counts[actual_class] += 1
        ap_sums[actual_class] += ap

        if i % 50 == 0:
          print_maps(ap_sums=ap_sums, ap_counts=ap_counts, class_map=class_map)
        sys.stdout.flush()

      print_maps(ap_sums=ap_sums, ap_counts=ap_counts, class_map=class_map)


if __name__ == '__main__':
  dataset_root = '/usr/local/google/home/plakal/fsd12k/final/dataset'
  test_csv_path = os.path.join(dataset_root, 'eval', 'dataset_test.csv')
  test_clip_dir = os.path.join(dataset_root, 'eval', 'audio')
  class_map_path = '/usr/local/google/home/plakal/fsd12k/baseline/class_map.csv'
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
  checkpoint_path = sys.argv[1]  # '/usr/local/google/home/plakal/fsd12k/baseline/train/model.ckpt-1001'

  eval(model_name=model_name, hparams=hparams, test_csv_path=test_csv_path,
        test_clip_dir=test_clip_dir, class_map_path=class_map_path,
       checkpoint_path=checkpoint_path)
