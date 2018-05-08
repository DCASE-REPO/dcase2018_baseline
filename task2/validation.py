import csv
from collections import defaultdict
import os
import random
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

def validate(model_name=None, hparams=None, validation_csv_path=None, validation_clip_dir=None,
             class_map_path=None, checkpoint_path=None):
  print('\nValidation for model:{} with hparams:{} and class map:{}'.format(model_name, hparams, class_map_path))
  print('Validation data: clip dir {} and labels {}'.format(validation_clip_dir, validation_csv_path))
  print('Checkpoint: {}\n'.format(checkpoint_path))

  with tf.Graph().as_default():
    label_class_index_table, num_classes = input.get_class_map(class_map_path)
    csv_record = tf.placeholder(tf.string, [])
    features, labels = input.record_to_labeled_log_mel_examples(
        csv_record, clip_dir=validation_clip_dir, hparams=hparams,
        label_class_index_table=label_class_index_table, num_classes=num_classes)
    global_step, prediction, _, _ = model.define_model(
        model_name=model_name, features=features, num_classes=num_classes,
        hparams=hparams, training=False)

    saver = tf.train.Saver()

    with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=checkpoint_path) as sess:
      class_map = {int(row[0]): row[1] for row in csv.reader(open(class_map_path))}
      ap_counts = defaultdict(int)
      ap_sums = defaultdict(float)
      validation_records = open(validation_csv_path).readlines()[1:]
      random.shuffle(validation_records)
      for (i,record) in enumerate(validation_records):
        record = record.strip()
        actual, predicted = sess.run([labels, prediction], {csv_record: record})

        actual_class = np.argmax(actual[0])
        predicted = np.average(predicted, axis=0)
        predicted_classes = np.argsort(predicted)[::-1][:TOP_N]
        ap = avg_precision(actual=actual_class, predicted=predicted_classes)
        print(class_map[actual_class], [class_map[index] for index in predicted_classes], ap)

        ap_counts[actual_class] += 1
        ap_sums[actual_class] += ap

        if i % 50 == 0:
          print_maps(ap_sums=ap_sums, ap_counts=ap_counts, class_map=class_map)
        sys.stdout.flush()

      print_maps(ap_sums=ap_sums, ap_counts=ap_counts, class_map=class_map)
