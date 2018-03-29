#!/usr/bin/env python

import os
import sys

import tensorflow as tf

import input
import model

def train(model_name=None, hparams=None, train_csv_path=None, train_clip_dir=None,
          class_map_path=None):
  with tf.Graph().as_default():
    features, labels, num_classes, input_init = input.train_input(
        train_csv_path=train_csv_path, train_clip_dir=train_clip_dir, class_map_path=class_map_path,
        hparams=hparams)
    global_step, prediction, loss_tensor, train_op = model.define_model(
        model_name=model_name, features=features, labels=labels, num_classes=num_classes,
        hparams=hparams, training=True)

    saver = tf.train.Saver(
        max_to_keep=30, keep_checkpoint_every_n_hours=0.25)
    saver_hook = tf.train.CheckpointSaverHook(
        save_steps=250, checkpoint_dir='train', saver=saver)

    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(
        save_steps=50, output_dir='train', summary_op=summary_op)

    with tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook]) as sess:
      sess.raw_session().run(input_init)
      while not sess.should_stop():
        step, _, loss = sess.run([global_step, train_op, loss_tensor])
        print(step, loss)
        sys.stdout.flush()

if __name__ == '__main__':
  dataset_root = '/usr/local/google/home/plakal/fsd12k/final/dataset'
  train_csv_path = os.path.join(dataset_root, 'dev', 'dataset_train.csv')
  train_clip_dir = os.path.join(dataset_root, 'dev', 'audio')
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

  train(model_name=model_name, hparams=hparams, train_csv_path=train_csv_path,
        train_clip_dir=train_clip_dir, class_map_path=class_map_path)
