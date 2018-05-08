#!/usr/bin/env python

import argparse
import sys
import tensorflow as tf

import inference
import train
import validation

def parse_flags():
  parser = argparse.ArgumentParser(description='DCASE 2018 Task 2 Baseline')

  # Flags common to all modes.
  all_modes_group = parser.add_argument_group('Flags common to all modes')
  all_modes_group.add_argument(
      '--mode', type=str, choices=['train', 'validation', 'inference'], required=True,
      help='Run one of training, validation, or inference.')
  all_modes_group.add_argument(
      '--model', type=str, choices=['cnn', 'mlp'], default='cnn', required=True,
      help='Name of a model architecture. Currently, one of "cnn" or "mlp".')
  all_modes_group.add_argument(
      '--hparams', type=str, default='',
      help='Model hyperparameters in comma-separated name=value format.')
  all_modes_group.add_argument(
      '--class_map_path', type=str, default='', required=True,
      help='Path to CSV file containing map between class index and name.')

  # Flags for training only.
  training_group = parser.add_argument_group('Flags for training only')
  training_group.add_argument(
      '--train_clip_dir', type=str, default='',
      help='Path to directory containing training clips.')
  training_group.add_argument(
      '--train_csv_path', type=str, default='',
      help='Path to CSV file containing training clip filenames and labels.')
  training_group.add_argument(
      '--train_dir', type=str, default='',
      help='Path to a directory which will hold model checkpoints and other outputs.')

  # Flags common to validation and inference.
  validation_inference_group = parser.add_argument_group('Flags for validation and inference')
  validation_inference_group.add_argument(
      '--checkpoint_path', type=str, default='',
      help='Path to a model checkpoint to use for validation or inference.')

  # Flags for validation only.
  validation_group = parser.add_argument_group('Flags for validation only')
  validation_group.add_argument(
      '--validation_clip_dir', type=str, default='',
      help='Path to directory containing validation clips.')
  validation_group.add_argument(
      '--validation_csv_path', type=str, default='',
      help='Path to CSV file containing validation clip filenames and labels.')

  # Flags for inference only.
  inference_group = parser.add_argument_group('Flags for inference only')
  inference_group.add_argument(
      '--test_clip_dir', type=str, default='',
      help='Path to directory containing test clips.')
  inference_group.add_argument(
      '--predictions_csv_path', type=str, default='',
      help='Path to a CSV file in which to store predictions.')

  flags = parser.parse_args()

  # Additional per-mode validation.
  try:
    if flags.mode == 'train':
      assert flags.train_clip_dir, 'Must specify --train_clip_dir'
      assert flags.train_csv_path, 'Must specify --train_csv_path'
      assert flags.train_dir, 'Must specify --train_dir'

    elif flags.mode == 'validation':
      assert flags.checkpoint_path, 'Must specify --checkpoint_path'
      assert flags.validation_clip_dir, 'Must specify --validation_clip_dir'
      assert flags.validation_csv_path, 'Must specify --validation_csv_path'

    else:
      assert flags.mode == 'inference'
      assert flags.checkpoint_path, 'Must specify --checkpoint_path'
      assert flags.test_clip_dir, 'Must specify --test_clip_dir'
      assert flags.predictions_csv_path, 'Must specify --predictions_csv_path'
  except AssertionError as e:
    print('\nError: ', e, '\n')
    parser.print_help()
    sys.exit(1)

  return flags

def parse_hparams(flag_hparams):
  # Default values for all hyperparameters.
  hparams = tf.contrib.training.HParams(
      # Window and hop length for Short-Time Fourier Transform applied to audio
      # waveform to make the spectrogram.
      stft_window_seconds=0.025,
      stft_hop_seconds=0.010,
      # Parameters controlling conversion of spectrogram into mel spectrogram.
      mel_bands=64,
      mel_min_hz=125,
      mel_max_hz=7500,
      # log mel spectrogram = Log(mel-spectrogram + mel_log_offset)
      mel_log_offset=0.001,
      # Window and hop length used to frame the log mel spectrogram into
      # examples.
      example_window_seconds=0.250,
      example_hop_seconds=0.125,
      # Number of examples in each batch fed to the model.
      batch_size=64,
      # For the 'mlp' multi-layer perceptron, nl=# layers, nh=# units per layer.
      nl=2,
      nh=256,
      # Standard-deviation of the normal distribution with mean 0 used to
      # initialize the weights of the model. Biases are initialized to 0.
      weights_init_stddev=1e-3,
      # Learning rate.
      lr=1e-4,
      # Epsilon passed to the Adam optimizer.
      adam_eps=1e-8,
      # Classifier layer: one of softmax or logistic.
      classifier='softmax')

  # Let flags override default hparam values.
  hparams.parse(flag_hparams)

  return hparams

def main():
  flags = parse_flags()
  hparams = parse_hparams(flags.hparams)

  if flags.mode == 'train':
    train.train(model_name=flags.model, hparams=hparams,
                class_map_path=flags.class_map_path,
                train_csv_path=flags.train_csv_path,
                train_clip_dir=flags.train_clip_dir,
                train_dir=flags.train_dir)

  elif flags.mode == 'validation':
    validation.validate(model_name=flags.model, hparams=hparams,
                        class_map_path=flags.class_map_path,
                        validation_csv_path=flags.validation_csv_path,
                        validation_clip_dir=flags.validation_clip_dir,
                        checkpoint_path=flags.checkpoint_path)

  else:
    assert flags.mode == 'inference'
    inference.predict(model_name=flags.model, hparams=hparams,
                      class_map_path=flags.class_map_path,
                      test_clip_dir=flags.test_clip_dir,
                      checkpoint_path=flags.checkpoint_path,
                      predictions_csv_path=flags.predictions_csv_path)

if __name__ == '__main__':
  main()
