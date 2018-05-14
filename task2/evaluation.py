"""Evaluation for DCASE 2018 Task 2 Baseline models."""

import csv
from collections import defaultdict
import os
import random
import sys

import numpy as np
import tensorflow as tf

import inputs
import model

# For Kaggle evaluation, we want MAP@3.
TOP_N = 3

def get_top_predicted_classes(predicted):
  """Computes the top N predicted classes given the prediction scores for all examples in a clip."""
  # For prediction, we average the prediction scores for each example in
  # the batch, and then take the indices of the top N by score.
  predicted = np.average(predicted, axis=0)
  predicted_classes = np.argsort(predicted)[::-1][:TOP_N]
  return predicted_classes

def avg_precision(actual=None, predicted=None):
  """Computes average label precision."""
  for (i, p) in enumerate(predicted):
    if actual == p:
      return 1.0 / (i + 1.0)
  return 0.0

def print_maps(ap_sums=None, ap_counts=None, class_map=None):
  """Prints per-class and overall MAP using running per-class sums/counts of AP."""
  map_count = 0
  map_sum = 0.0
  print('\n')
  for class_index in sorted(ap_counts.keys()):
    m_ap = ap_sums[class_index] / ap_counts[class_index]
    print('MAP for %s: %.4f' % (class_map[class_index], m_ap))
    map_count += ap_counts[class_index]
    map_sum += ap_sums[class_index]
  m_ap = map_sum / map_count
  print('Overall MAP: %.4f\n' % m_ap)

def evaluate(model_name=None, hparams=None, eval_csv_path=None, eval_clip_dir=None,
             class_map_path=None, checkpoint_path=None):
  """Runs the evaluation loop."""
  print('\nEvaluation for model:{} with hparams:{} and class map:{}'.format(model_name, hparams, class_map_path))
  print('Evaluation data: clip dir {} and labels {}'.format(eval_clip_dir, eval_csv_path))
  print('Checkpoint: {}\n'.format(checkpoint_path))

  with tf.Graph().as_default():
    label_class_index_table, num_classes = inputs.get_class_map(class_map_path)
    csv_record = tf.placeholder(tf.string, [])  # fed during evaluation loop.

    # Use a simpler in-order input pipeline for eval than the one used in
    # training, since we don't want to shuffle examples across clips.
    # The features consist of a batch of all possible framed log mel spectrum
    # examples from the same clip. The labels in this case will contain a batch
    # of identical 1-hot vectors.
    features, labels = inputs.record_to_labeled_log_mel_examples(
        csv_record, clip_dir=eval_clip_dir, hparams=hparams,
        label_class_index_table=label_class_index_table, num_classes=num_classes)

    # Create the model in prediction mode.
    global_step, prediction, _, _ = model.define_model(
        model_name=model_name, features=features, num_classes=num_classes,
        hparams=hparams, training=False)

    with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=checkpoint_path) as sess:
      # Read in class map CSV into a class index -> class name map.
      class_map = {int(row[0]): row[1] for row in csv.reader(open(class_map_path))}

      # Keep running counters to aid printing incremental AP stats.
      ap_counts = defaultdict(int)  # maps class index to the number of clips with that label.
      ap_sums = defaultdict(float)  # maps class index to the sum of AP for all clips with that label.

      # Read in the validation CSV, skippign the header.
      eval_records = open(eval_csv_path).readlines()[1:]
      # Shuffle the lines so that as we print incremental stats, we get good
      # coverage across classes and get a quick initial impression of how well
      # the model is doing even before evaluation is completely done.
      random.shuffle(eval_records)

      for (i,record) in enumerate(eval_records):
        record = record.strip()
        actual, predicted = sess.run([labels, prediction], {csv_record: record})

        # By construction, actual consists of identical rows, where each row is
        # the same 1-hot label (because we are looking at features from the same
        # clip). np.argmax() of any of the rows (and in particular [0]) will
        # give us the class index corresponding to the label.
        actual_class = np.argmax(actual[0])

        predicted_classes = get_top_predicted_classes(predicted)

        # Compute AP for this item, update running counters/sums, and print the
        # prediction vs actual.
        ap = avg_precision(actual=actual_class, predicted=predicted_classes)
        ap_counts[actual_class] += 1
        ap_sums[actual_class] += ap
        print(class_map[actual_class], [class_map[index] for index in predicted_classes], ap)

        # Periodically print per-class and overall AP stats.
        if i % 50 == 0:
          print_maps(ap_sums=ap_sums, ap_counts=ap_counts, class_map=class_map)
        sys.stdout.flush()

      # Final AP stats.
      print_maps(ap_sums=ap_sums, ap_counts=ap_counts, class_map=class_map)
