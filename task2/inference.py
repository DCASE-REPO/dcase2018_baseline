import csv
from collections import defaultdict
import os
import sys

import numpy as np
import tensorflow as tf

import input
import model

TOP_N = 3

def predict(model_name=None, hparams=None, test_clip_dir=None,
            class_map_path=None, checkpoint_path=None, predictions_csv_path=None):
  print('\nInference for model:{} with hparams:{} and class map:{}'.format(model_name, hparams, class_map_path))
  print('Inference data: clip dir {} and checkpoint {}'.format(test_clip_dir, checkpoint_path))
  print('Predictions in CSV {}\n'.format(predictions_csv_path))

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
      test_clips = sorted(os.listdir(test_clip_dir))[:10]
      pred_writer = csv.DictWriter(open(predictions_csv_path, 'w'), fieldnames=['fname', 'label'])
      pred_writer.writeheader()
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
