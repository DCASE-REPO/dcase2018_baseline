
import tensorflow as tf

slim = tf.contrib.slim

def define_mlp(features=None, hparams=None, training=False):
  net = slim.flatten(features)
  net = slim.repeat(net, hparams.nl, slim.fully_connected, hparams.nh)
  return net

def define_cnn(features=None, hparams=None, training=False):
  net = tf.expand_dims(features, axis=3)
  with slim.arg_scope([slim.conv2d],
                      stride=1, padding='SAME'), \
       slim.arg_scope([slim.max_pool2d],
                      stride=2, padding='SAME'):
    net = slim.conv2d(net, 100, kernel_size=[7, 7], stride=1)
    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2)
    net = slim.conv2d(net, 150, kernel_size=[5, 5], stride=1)
    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2)
    net = slim.conv2d(net, 200, kernel_size=[3, 3], stride=1)
    net = slim.max_pool2d(net, kernel_size=[12, 8], stride=1)
    net = slim.flatten(net)
  return net

def define_model(model_name=None, features=None, labels=None, num_classes=None, hparams=None,
                 training=False):
  global_step = tf.Variable(
      0, name='global_step', trainable=training,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                   tf.GraphKeys.GLOBAL_STEP])

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
                      #weights_initializer=slim.initializers.xavier_initializer(),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      trainable=training):
    if model_name == 'mlp':
      embedding = define_mlp(features=features, hparams=hparams, training=training)
    elif model_name == 'cnn':
      embedding = define_cnn(features=features, hparams=hparams, training=training)
    else:
      raise ValueError('Unknown model %s' % model)

    logits = slim.fully_connected(embedding, num_classes)
    prediction = tf.nn.softmax(logits)

  if training:
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
