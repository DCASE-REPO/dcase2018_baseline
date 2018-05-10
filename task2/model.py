"""Model definitions for DCASE 2018 Task 2 Baseline models."""

import csv

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def define_mlp(features=None, hparams=None):
  """Defines a multi-layer perceptron model, without the classifier layer."""
  net = slim.flatten(features)
  net = slim.repeat(net, hparams.nl, slim.fully_connected, hparams.nh)
  return net

def define_cnn(features=None, hparams=None):
  """Defines a convolutional neural network model, without the classifier layer."""
  net = tf.expand_dims(features, axis=3)
  with slim.arg_scope([slim.conv2d],
                      stride=1, padding='SAME'), \
       slim.arg_scope([slim.max_pool2d],
                      stride=2, padding='SAME'):
    net = slim.conv2d(net, 100, kernel_size=[7, 7])
    net = slim.max_pool2d(net, kernel_size=[3, 3])
    net = slim.conv2d(net, 150, kernel_size=[5, 5])
    net = slim.max_pool2d(net, kernel_size=[3, 3])
    net = slim.conv2d(net, 200, kernel_size=[3, 3])
    net = tf.reduce_max(net, axis=[1,2], keepdims=True)
    net = slim.flatten(net)
  return net

def define_model(model_name=None, features=None, labels=None, num_classes=None, hparams=None,
                 training=False):
  """Defines a classifier model.

  Args:
    model_name: one of ['mlp', 'cnn'], determines the model architecture.
    features: tensor containing a batch of input features.
    labels: tensor containing a batch of corresponding labels.
    num_classes: number of classes.
    hparams: model hyperparameters.
    training: True iff the model is being trained.

  Returns:
    global_step: tensor containing the global step.
    prediction: tensor containing the predictions from the classifier layer.
    loss: tensor containing the training loss for each batch.
    train_op: op that runs training (forward and backward passes) for each batch.
  """
  global_step = tf.Variable(
      0, name='global_step', trainable=training,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                   tf.GraphKeys.GLOBAL_STEP])

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(stddev=hparams.weights_init_stddev),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      trainable=training):
    # Define the model without the classifier layer.
    if model_name == 'mlp':
      embedding = define_mlp(features=features, hparams=hparams)
    elif model_name == 'cnn':
      embedding = define_cnn(features=features, hparams=hparams)
    else:
      raise ValueError('Unknown model %s' % model)

    # Add the logits and the classifier layer.
    logits = slim.fully_connected(embedding, num_classes, activation_fn=None)
    if hparams.classifier == 'logistic':
      prediction = tf.nn.sigmoid(logits)
    elif hparams.classifier == 'softmax':
      prediction = tf.nn.softmax(logits)
    else:
      raise ValueError('Bad classifier: %s' % classifier)

  if training:
    # In training mode, also create loss and train op.
    if hparams.classifier == 'logistic':
      xent = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=labels)
    elif hparams.classifier == 'softmax':
      xent = tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=logits, labels=labels)

    loss = tf.reduce_mean(xent)
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=hparams.lr,
        epsilon=hparams.adam_eps)
    train_op = optimizer.minimize(loss, global_step=global_step)
  else:
    loss = None
    train_op = None

  return global_step, prediction, loss, train_op
