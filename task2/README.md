# Baseline system for Task 2 of DCASE 2018

This is the baseline system for [Task 2](http://dcase.community/challenge2018/task-general-purpose-audio-tagging) of the
[DCASE 2018](http://dcase.community/challenge2018) challenge. The system implements an audio classifier using a simple
convolutional neural network, which takes log mel spectrogram features as input and produces ranked predictions over
the 41 classes in the dataset.

Task 2 is hosted as a [Kaggle challenge](https://kaggle.com/c/freesound-audio-tagging)
and the baseline system was used to produce the baseline submission named _Challenge Baseline: Log Mel Spectrogram
features, 3-layer CNN_ in the [Kaggle challenge leaderboard](https://kaggle.com/c/freesound-audio-tagging/leaderboard).

## Installation

* Clone [this GitHub repository](https://github.com/DCASE-REPO/dcase2018_baseline). Task 2's code is inside the `task2` directory.
* Dependence version requirements: python >= 3.5.3, tensorflow >= 1.6.0, numpy >= 1.14.2. The baseline was tested on a
  machine running a Debian-like Linux OS, but should be portable to other OSes.
* Download the dataset [from Kaggle](https://kaggle.com/c/freesound-audio-tagging/data): `audio_train.zip`, `audio_test.zip`, `train.csv`. Unzip the zip files to produce `audio_train` and `audio_test` directories containing audio clips for training and testing, respectively.

## Code Layout

* `main.py`: Main driver. Run main.py --help to see all available flags.
* `train.py`: Training loop. Called by `main.py` when passed `--mode train`
* `evaluation.py`: Computes evaluation metrics. Called by `main.py` when passed `--mode evaluation`
* `inference.py`: Generates model predictions. Called by `main.py` when passed `--mode inference`
* `inputs.py`: TensorFlow input pipeline for decoding CSV input and WAV files, and constructing
   framed and labeled log mel spectrogtram examples.
* `model.py`: Tensorflow model definitions.
* `make_class_map.py`: Utility to create a class map from the training dataset.

## Usage

* If you want to use a validation set to compare models, prepare a hold-out validation set by moving some random
  fraction (say, 10%) of the rows from `train.csv` into `validation.csv`, while keeping the same header line.

* Prepare a class map, which is a CSV file that maps between class indices and class names, and is used by various parts
  of the system:
```shell
$  make_class_map.py < /path/to/train.csv > /path/to/class_map.csv
```

* Train a CNN model with checkpoints created in `train_dir`:
```shell
$ main.py \
    --mode train \
    --model cnn \
    --class_map_path /path/to/class_map.csv \
    --train_clip_dir /path/to/audio_train \
    --train_csv_path /path/to/train.csv \
    --train_dir /path/to/train_dir
```
  This will produce checkpoint files in `train_dir` having the name prefix `model.ckpt-N` with increasing N, where N
  represents the number of batches of examples seen by the model.

  This will also produce a log of the loss at each step on standard output, as well as a TensorFlow event log in `train_dir`
  which can be viewed by running a TensorBoard dashboard pointed at that directory.

  By default, this will use the default hyperparameters defined inside `main.py`. These can be overridden using the
  `--hparams` flag to pass in comma-separated `name=value` pairs. For example, `--hparams batch_size=32,lr=0.01` will
  use a batch size of 32 and a learning rate of 0.01. For more information about the hyperparameters, see below in the
  Model description section. Note that if you use non-default hyperparameters during training, you must use the same
  hyperparameters when running the evaluation and inference steps described below.

* Evaluate a particular trained model checkpoint on the validation set:
```shell
$ main.py \
    --mode eval \
    --model cnn \
    --class_map_path /path/to/class_map.csv \
    --eval_clip_dir /path/to/audio_train \
    --eval_csv_path /path/to/validation.csv \
    --checkpoint /path/to/train_dir/model.ckpt-<N>
```
  This will produce a per-class MAP as well as overall MAP report on standard output.

* Generate predictions in `predictions.csv` from a particular trained model checkpoint for submission to Kaggle:
```shell
$ main.py \
    --mode inference \
    --model cnn \
    --class_map_path /path/to/class_map.csv \
    --test_clip_dir /path/to/audio_test \
    --checkpoint /path/to/train_dir/model.ckpt-<N> \
    --predictions_csv_path /path/to/predictions.csv
```

## Model Description and Performance

The baseline system implements a convolutional neural network (CNN) classifier similar to, but scaled down from, the
deep CNN models that have been very successful in the vision domain. The model takes framed examples of log mel
spectrogram as input and produces ranked predictions over the 41 classes in the dataset.

The baseline system also allows training a simpler fully connected multi-layer perceptron (MLP) classifier, which
can be selected by passing in the flag `--model mlp`.

### Input features

We use frames of log mel spectrogram as input features, which has been demonstrated to work well for audio CNN classifiers by
Hershey et. al. [1].

We compute log mel spectrogram examples as follows:

* The incoming audio is always at 44.1 kHz mono.

* The [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) is computed using the magnitude of the [Short-Time Fourier Transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) (STFT) with a window size of 25ms,
  a window hop size of 10ms, and a periodic Hann window.

* The mel spectrogram is computed by mapping the spectrogram to 64 mel bins covering the range 125-7500 Hz. The [mel scale](https://en.wikipedia.org/wiki/Mel_scale)
  is intended to better represent human audio perception by using more bins in the lower frequencies and fewer bins
  in the higher frequencies.

* The stabilized log mel spectrogram is computed by applying `log`(mel spectrogram + 0.001) where the offset of 0.01 is used
  to avoid taking a logarithm of 0. The compressive non-linearity of the logarithm is used to reduce the dynamic range
  of the feature values.

* The log mel spectrogram is then framed into overlapping examples with a window size of 0.25s and a hop size of 0.125s.
  The overlap allows generating more examples from the same data than with no overlap, which helps to increase the
  effective size of the dataset, and also gives the model a little more context to learn from because it now sees the
  same slice of audio in several different examples with varying prefixes and suffixes. The window size was chosen to
  be small enough that we could generate an example from even the smallest clip in the dataset (~0.3s), and yet large
  enough that it offers the model enough context.

The input pipeline parses CSV records, decodes WAV files, creates examples containing log mel spectrum examples with
1-hot-encoded labels, shuffles them across clips, and does all of this on-the-fly and purely in TensorFlow, without
requiring any Python preprocessing or separate feature generation or storage step.

### Architecture

The baseline CNN model consists of three 2-D convolutional layers (with ReLU activations) and alternating 2-D max-pool
layers, followed by a final max-reduction (to produce a single value per feature map), and a softmax layer. The Adam
optimizer is used to train the model.

The layers are listed in the table below using notation Conv2D(kernel size, stride, # feature maps) and MaxPool2D(kernel size,
stride). Both Conv2D and MaxPool2D use the `SAME` padding scheme. ReduceMax applies a maximum-value reduction across the
first two dimensions. Activation shapes do not include the batch dimension.

 Layer              | Activation shape | # Weights | # Multiplies
--------------------|------------------|----------:|---------------:
Input               | (25, 64, 1)      | 0         | 0
Conv2D(7x7, 1, 100) | (25. 64. 100)    | 4.9K      | 7.8M
MaxPool2D(3x3, 2x2) | (13, 32, 100)    | 0         | 0
Conv2D(5x5, 1, 150) | (13, 32, 150)    | 375K      | 156M
MaxPool2D(3x3, 2x2) | (7, 16, 150)     | 0         | 0
Conv2D(3x3, 1, 200) | (7, 16, 200)     | 270K      | 30.2M
ReduceMax           | (1, 1, 200)      | 0         | 0
Softmax             | (41,)            | 8.2K      | 8.2K
**Total**           |                  | **658.1K**| **194.1M**

### Hyperparameters

The following hyperparameters, defined with their default values in `main.py`,
are used in the input pipeline and model definition.

```python
hparams = tf.contrib.training.HParams(
    # Window and hop length for Short-Time Fourier Transform applied to audio
    # waveform to make the spectrogram.
    stft_window_seconds=0.025,
    stft_hop_seconds=0.010,
    # Parameters controlling conversion of spectrogram into mel spectrogram.
    mel_bands=64,
    mel_min_hz=125,
    mel_max_hz=7500,
    # log mel spectrogram = log(mel-spectrogram + mel_log_offset)
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
    # Standard deviation of the normal distribution with mean 0 used to
    # initialize the weights of the model. Biases are initialized to 0.
    weights_init_stddev=1e-3,
    # Learning rate.
    lr=1e-4,
    # Epsilon passed to the Adam optimizer.
    adam_eps=1e-8,
    # Classifier layer: one of softmax or logistic.
    classifier='softmax')
```

In order to override the defaults, pass the `--hparams` flag a comma-separated
list of `name=value` pairs.  For example,
`--hparams example_window_seconds=0.5,batch_size=32,lr=0.01` will use examples
of size 0.5s, a batch size of 32, and a learning rate of 0.01.

### Clip Prediction

The classifier predicts 41 scores for individual 0.25s-wide examples. In order
to produce a ranked list of predicted classes for an entire clip, we average the
predictions from all framed examples generated from the clip, and take the top 3
classes by score.

### Performance

The baseline system trains to achieve an MAP@3 of ~0.7 on the public Kaggle leaderboard after ~5 epochs of the entire
training set which are completed in ~12 hours on an
[`n1-standard-8`](https://cloud.google.com/compute/docs/machine-types#standard_machine_types) Google Compute Engine machine with a quad-core Intel Xeon E5 v3 (Haswell) @ 2.3 GHz.

An aside on computing epoch sizes: a simple back-of-the-envelope calculation uses the fact that we use uncompressed WAVs
with a fixed sample rate (44.1 kHz) and a fixed sample size (16-bit signed PCM). The total size of the `audio_train`
directory containing all the clips is 5.4 GB. Each sample is 2 bytes, and each second needs 44100 samples, so the total
number of seconds in the training set is (5.4 * 2 ^ 30) / (2 * 44100) = ~65739. We frame examples with a hop of 0.125
seconds and we use 64 examples in a batch, so an epoch of all examples in the training set consists of 65739 / 0.125 /
64 = ~8217 batches. Letting the model train for 5 epochs would mean 8217 * 5 = ~41K steps.

Note that we did not perform any hyperparameter or architectural tuning of the baseline with a validation set. We picked
the architecture and the default values of hyperparameters based on our experience with training similar models in the
Sound Understanding team at Google, and they happened to work well for this task.

Also note that the input pipeline defined in `inputs.py` contains some fixed parameters that affect training speed
(parallelism of example extraction, various buffer sizes, prefetch parameters, etc) and which should be changed
if the available machine resources don't match what we used to build the baseline.

## Ideas for improvement

* Try a deeper architecture, such as Inception or ResNet, as explored by Hershey et. al. [1].

* Try tuning the hyperparameters with a validation set.

* Try the usual tricks used for deep networks on small datasets: dropout, weight decay, etc.

* Try giving more context to the model. The classifier currently sees each example in isolation and we performs a simple
  averaging of all the individual predictions, but the model could potentially do a better job if it could use
  information about what else is happening in the clip. One way to do this is to use a longer context window (but now you
  have to pad short clips to fill an example). Another way is to switch to a recurrent model and explicitly learn how to
  combine information from various parts of the same clip.

* Look at the per-class metrics produced by the evaluation and try to figure out how to improve the worst performing
  classes. E.g., you might have to take the class occurrence prior into account when computing the loss, so that the
  model tries harder to get the rarer classes correct. Or you might have to treat the manually verified labels
  differently from unverified labels.

* Try pretraining the model using a large collection of weakly labelled environmental sounds, e.g.,
  [AudioSet](http://g.co/audioset). Then fine-tune this pretrained model on the challenge dataset.

* Try feeding raw audio waveform to the model instead of log mel spectrum features.

## Contact

For general discussion of this task, please use the [Kaggle Discussion board](https://www.kaggle.com/c/freesound-audio-tagging/discussion).

For specific issues with the code for this baseline system, please create an issue or a pull request on GitHub for the
[DCASE 2018 Baseline repo](https://github.com/DCASE-REPO/dcase2018_baseline) and make sure to @-mention `plakal`.

## References

1. Hershey, S. et. al., [CNN Architectures for Large-Scale Audio Classification](https://ai.google/research/pubs/pub45611), ICASSP 2017.
