"""Trainer for DCASE 2018 Task 2 Baseline models."""

import os
import sys

import tensorflow as tf

import inputs
import model

def train(model_name=None, hparams=None, train_csv_path=None, train_clip_dir=None,
          class_map_path=None, train_dir=None):
  """Runs the training loop."""
  print('\nTraining model:{} with hparams:{} and class map:{}'.format(model_name, hparams, class_map_path))
  print('Training data: clip dir {} and labels {}'.format(train_clip_dir, train_csv_path))
  print('Training dir {}\n'.format(train_dir))

  with tf.Graph().as_default():
    # Create the input pipeline.
    features, labels, num_classes, input_init = inputs.train_input(
        train_csv_path=train_csv_path, train_clip_dir=train_clip_dir, class_map_path=class_map_path,
        hparams=hparams)
    # Create the model in training mode.
    global_step, prediction, loss_tensor, train_op = model.define_model(
        model_name=model_name, features=features, labels=labels, num_classes=num_classes,
        hparams=hparams, training=True)

    # Define our own checkpoint saving hook, instead of using the built-in one,
    # so that we can specify additional checkpoint retention settings.
    saver = tf.train.Saver(
        max_to_keep=30, keep_checkpoint_every_n_hours=0.25)
    saver_hook = tf.train.CheckpointSaverHook(
        save_steps=250, checkpoint_dir=train_dir, saver=saver)

    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(
        save_steps=50, output_dir=train_dir, summary_op=summary_op)

    with tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook],
                                           checkpoint_dir=train_dir) as sess:
      sess.raw_session().run(input_init)
      while not sess.should_stop():
        step, _, pred, loss = sess.run([global_step, train_op, prediction, loss_tensor])
        print(step, loss)
        sys.stdout.flush()
