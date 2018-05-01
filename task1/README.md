DCASE2018 - Task 1 - Baseline systems
-------------------------------------

Author:

**Toni Heittola**, *Tampere University of Technology* [Email](mailto:toni.heittola@tut.fi), [Homepage](http://www.cs.tut.fi/~heittolt/), [GitHub](https://github.com/toni-heittola)

Getting started
===============

1. Clone repository from [Github](https://github.com/DCASE-REPO/dcase2018_baseline). The baseline code for Task 1 is available under subdirectory `task1`.
2. Install requirements with command: `pip install -r requirements.txt`.
3. Run the subtask specific applications with default settings:
   - Subtask A: `python task1a.py` or  `./task1a.py`
   - Subtask B: `python task1b.py` or  `./task1b.py`

Introduction
============

This is the baseline system for the [Acoustic scene classification task (Task 1)](http://dcase.community/challenge2018/task-acoustic-scene-classification) in Detection and Classification of Acoustic Scenes and Events 2018 (DCASE2018) challenge. The system is intended to provide a simple entry-level state-of-the-art approach that gives reasonable results in the subtasks of Task 1. The baseline system is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox. 

Participants are strongly encouraged to build their own systems by extending the provided baseline system. The system has all needed functionality for the dataset handling, acoustic feature storing and accessing, acoustic model training and storing, and evaluation. The modular structure of the system enables participants to modify the system to their needs. The baseline system is a good starting point especially for the entry level researchers to familiarize themselves with the acoustic scene classification problem.

If participants plan to publish their code to the DCASE community after the challenge, building their approach on the baseline system will make their code more accessible to the community. DCASE organizers strongly encourage participants to share their code in any form after the challenge.

Subtasks
=========

### Subtask A - Acoustic Scene Classification

[TUT Urban Acoustic Scenes 2018 Development dataset](https://zenodo.org/record/1228142) is used as development dataset for this task.

This subtask is concerned with the basic problem of acoustic scene classification, in which all available data (development and evaluation) are recorded with the same device, in this case device A. The dataset contains 864 10-second segments (48 kHz / 24bit / stereo) for each acoustic scene (144 minutes of audio). The dataset contains in total 8640 segments, i.e. 24 hours of audio. For a more detailed description see [DCASE Challenge task description](http://dcase.community/challenge2018/task-acoustic-scene-classification).

The subtask specific baseline system is implemented in file `task1a.py`.

#### Results for development dataset

The cross-validation setup provided with the *TUT Urban Acoustic Scenes 2018 Development dataset* is used to evaluate the performance of the baseline system. Results are calculated using TensorFlow in GPU mode (using Nvidia Titan XP GPU card). Because results produced with GPU card are generally non-deterministic, the system was trained and tested 10 times, and mean and standard deviation of the performance from these 10 independent trials are shown in the results tables.

| Scene class        | Accuracy             |
| ------------------ | -------------------- |
| Airport            | 72.9 %               |
| Bus                | 62.9 %               |
| Metro              | 51.2 %               |
| Metro station      | 55.4 %               |
| Park               | 79.1 %               |
| Public square      | 40.4 %               |
| Shopping mall      | 49.6 %               |
| Street, pedestrian | 50.0 %               |
| Street, traffic    | 80.5 %               |
| Tram               | 55.1 %               |
| **Average**        | **59.7 %** (+/- 0.7) |

**Note:** The reported system performance is not exactly reproducible due to varying setups. However, you should be able obtain very similar results.


### Subtask B - Acoustic Scene Classification with mismatched recording devices

[TUT Urban Acoustic Scenes 2018 Mobile Development dataset](https://zenodo.org/record/1228235) is used as development
dataset for this task.

This subtask is concerned with the situation in which an application will be tested with a few different types of devices, possibly not the same as the ones used to record the development data. The dataset contains material recorded with devices A, B and C. For each acoustic scene there are 864 segments recorded with device A, and parallel audio consisting of 72 segments recorded with devices B and C. Data from device A was resampled and averaged into a single channel, to align with the properties of the data recorded with devices B and C (44.1 kHz / 24bit / mono). The dataset contains in total 28 hours of audio. For a more detailed description see [DCASE Challenge task description](http://dcase.community/challenge2018/task-acoustic-scene-classification).

The subtask specific baseline system is implemented in file `task1b.py`.

#### Results for development dataset

The cross-validation setup provided with the *TUT Urban Acoustic Scenes Mobile 2018 Development dataset* is used to evaluate the performance of the baseline system. Material from device A (high-quality) is used for training, and testing is done with material from all three devices. This highlights the problem of mismatched recording devices. Results are calculated the same way as for subtask A, with mean and standard deviation of the performance from 10 independent trials show in the results table.

| Scene class        | Device B             | Device C             | Average (B/C)        |      | Device A             |
| ------------------ | -------------------- | -------------------- | -------------------- | ---- |--------------------- |
| Airport            | 68.9 %               | 76.1 %               | 72.5 %               |      | 73.4 %               |
| Bus                | 70.6 %               | 86.1 %               | 78.3 %               |      | 56.7 %               |
| Metro              | 23.9 %               | 17.2 %               | 20.6 %               |      | 46.6 %               |
| Metro station      | 33.9 %               | 31.7 %               | 32.8 %               |      | 52.9 %               |
| Park               | 67.2 %               | 51.1 %               | 59.2 %               |      | 80.8 %               |
| Public square      | 22.8 %               | 26.7 %               | 24.7 %               |      | 37.9 %               |
| Shopping mall      | 58.3 %               | 63.9 %               | 61.1 %               |      | 46.4 %               |
| Street, pedestrian | 16.7 %               | 25.0 %               | 20.8 %               |      | 55.5 %               |
| Street, traffic    | 69.4 %               | 63.3 %               | 66.4 %               |      | 82.5 %               |
| Tram               | 18.9 %               | 20.6 %               | 19.7 %               |      | 56.5 %               |
| **Average**        | **45.1 %** (+/- 3.6) | **46.2 %** (+/- 4.2) | **45.6 %** (+/- 3.6) |      | **58.9 %** (+/- 0.8) |

**Note:** The reported system performance is not exactly reproducible due to varying setups. However, you should be able obtain very similar results.

### Subtask C - Acoustic Scene Classification with use of external data

This subtask is meant to test if use of external data in system development brings a significant improvement to the performance. The task is identical to subtask A, with the only difference that **use of external data and transfer learning is allowed** under certain conditions. More detailed description see [DCASE Challenge task description](http://dcase.community/challenge2018/task-acoustic-scene-classification).

As the subtask C is identical to A, there is no separate baseline system for it.

System description
==================

The system implements a convolutional neural network (CNN) based approach, where log mel-band energies are first extracted for each 10-second signal, and a network consisting of two CNN layers and one fully connected layer is trained to assign scene labels to the audio signals.

### Parameters

#### Acoustic features

- Analysis frame 40 ms (50% hop size)
- Log mel-band energies (40 bands)

#### Neural network

- Input shape: 40 * 500 (10 seconds)
- Architecture (definition for [task1A](https://github.com/DCASE-REPO/dcase2018_baseline/blob/master/task1/task1a.yaml#L141) and for [task1B](https://github.com/DCASE-REPO/dcase2018_baseline/blob/master/task1/task1b.yaml#L148)):
  - CNN layer #1
    - 2D Convolutional layer (filters: 32, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (5, 5)) + Dropout (rate: 30%)
  - CNN layer #2
    - 2D Convolutional layer (filters: 64, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (4, 100)) + Dropout (rate: 30%)
  - Flatten
  - Dense layer #1
    - Dense layer (units: 100, activation: ReLu )
    - Dropout (rate: 30%)
  - Output layer (activation: softmax)
- Learning (epochs: 200, batch size: 16, data shuffling between epochs)
  - Optimizer: Adam (learning rate: 0.001)
- Model selection:
  - Approximately 30% of the original training data is assigned to validation set, split done so that training and validation sets do not have segments from same location and so that both sets have similar amount of data per city
  - Model performance after each epoch is evaluated on the validation set, and best performing model is selected

**Network summary**

     _________________________________________________________________
     Layer (type)                 Output Shape              Param #   
     =================================================================
     conv2d_7 (Conv2D)            (None, 40, 500, 32)       1600      
     _________________________________________________________________
     batch_normalization_7 (Batch (None, 40, 500, 32)       160       
     _________________________________________________________________
     activation_7 (Activation)    (None, 40, 500, 32)       0         
     _________________________________________________________________
     max_pooling2d_7 (MaxPooling2 (None, 8, 100, 32)        0         
     _________________________________________________________________
     dropout_10 (Dropout)         (None, 8, 100, 32)        0         
     _________________________________________________________________
     conv2d_8 (Conv2D)            (None, 8, 100, 64)        100416    
     _________________________________________________________________
     batch_normalization_8 (Batch (None, 8, 100, 64)        32        
     _________________________________________________________________
     activation_8 (Activation)    (None, 8, 100, 64)        0         
     _________________________________________________________________
     max_pooling2d_8 (MaxPooling2 (None, 2, 1, 64)          0         
     _________________________________________________________________
     dropout_11 (Dropout)         (None, 2, 1, 64)          0         
     _________________________________________________________________
     flatten_4 (Flatten)          (None, 128)               0         
     _________________________________________________________________
     dense_7 (Dense)              (None, 100)               12900     
     _________________________________________________________________
     dropout_12 (Dropout)         (None, 100)               0         
     _________________________________________________________________
     dense_8 (Dense)              (None, 10)                1010      
     =================================================================
     Total params: 116,118
     Trainable params: 116,022
     Non-trainable params: 96
     _________________________________________________________________
    
     Input shape                     : (None, 40, 500, 1)
     Output shape                    : (None, 10)

Usage
=====

For each subtask there is a separate application (.py file):

- `task1a.py`, DCASE2018 baseline for Task 1A, Acoustic scene classification
- `task1b.py`, DCASE2018 baseline for Task 1B, Acoustic Scene Classification with mismatched recording devices


### Application arguments

All the usage arguments are shown by ``python task1a.py -h``.

| Argument                    |                                   | Description                                                  |
| --------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `-h`                        | `--help`                          | Application help.                                            |
| `-v`                        | `--version`                       | Show application version.                                    |
| `-m {dev,eval,leaderboard}` | `--mode {dev,eval,leaderboard}`   | Selector for application operation mode                      |
| `-s PARAMETER_SET`          | `--parameter_set PARAMETER_SET`   | Parameter set id. Can be also comma separated list e.g. `-s set1,set2,set3``. In this case, each set is run separately. |
| `-p FILE`                   | `--param_file FILE`               | Parameter file (YAML) to overwrite the default parameters    |
| `-o OUTPUT_FILE`            | `--output OUTPUT_FILE`            | Output file                                                  |
|                             | `--overwrite`                     | Force overwrite mode.                                        |
|                             | `--download_dataset DATASET_PATH` | Download dataset to given path and exit                      |
|                             | `--show_parameters`               | Show active application parameter set                        |
|                             | `--show_sets`                     | List of available parameter sets                             |
|                             | `--show_results`                  | Show results of the evaluated system setups                  |

### Operation modes

The system can be used in three different operation modes.

**Development mode** - `dev`

In development mode, the development dataset is used with the provided cross-validation setup: training set is used for learning, and testing set is used for evaluating the performance of the system. This is the default operation mode. 

Usage example: `python task1a.py` or `python task1a.py -m dev`

**Challenge mode** - `eval`

**Note:** This operation mode does not work yet as evaluation dataset has not been published yet. 

In challenge mode, the full development dataset (including training and test subsets) is used for learning, and a second dataset, evaluation dataset, is used for testing. The system system outputs are generated based on the evaluation dataset. If ground truth is available for the evaluation dataset, the output is also evaluated. This mode is designed to be used for generating the DCASE challenge submission, running the system on the evaluation dataset for generating the system outputs for the submission file. 

Usage example: `python task1a.py -m eval` and `python task1b.py -m eval`

To save system output to a file: `python task1a.py -m eval -o output.csv`

**Leaderboard mode** - `leaderboard`

**Note:** This operation mode does not work yet as the leaderboard dataset has not been published yet. 

Leaderboard mode is similar to challenge mode, except that instead of the official evaluation dataset, a specific leaderboard dataset is used. This mode can be used to produce the system output for the public leaderboard submission.

`python task1a.py -m leaderboard -o output.csv`

### System parameters

The baseline system supports multi-level parameter overwriting, to enable flexible switching between different system setups. Parameter changes are tracked with hashes calculated from parameter sections. These parameter hashes are used in the storage file paths when saving data (features, model, or results). By using this approach, the system will compute features, models and results only once for the specific parameter set, and after that it will reuse this precomputed data.

#### Parameter overwriting

Parameters are stored in YAML-formatted files, which are handled internally in the system as Dict like objects (`dcase_util.containers.DCASEAppParameterContainer`). **Default parameters** is the set of all possible parameters recognized by the system. **Parameter set** is a smaller set of parameters used to overwrite values of the default parameters. This can be used to select methods for processing, or tune parameters.

#### Parameter file

Parameters files are YAML-formatted files, containing the following three blocks:

- `active_set`, default parameter set id
- `sets`, list of dictionaries
- `defaults`, dictionary containing default parameters which are overwritten by the `sets[active_set]`

At the top level of the parameter dictionary there are parameter sections; depending on the name of the section, the parameters inside it are processed sometimes differently. Usually there is a main section (`feature_extractor`, and method parameter section (`feature_extractor_method_parameters`) which contains parameters for each possible method. When parameters are processed, the correct method parameters are copied from method parameter section to the main section under parameters. This allows having many methods ready parametrized and easily accessible.

#### Parameter hash

Parameter hashes are MD5 hashes calculated for each parameter section. In order to make these hashes more robust, some pre-processing is applied before hash calculation:

- If section contains field `enable` with value `False`, all fields inside this section are excluded from the parameter hash calculation. This will avoid recalculating the hash if the section is not used but some of these unused parameters are changed.
- If section contains fields with value `False`, these fields are excluded from the parameter hash calculation. This will enable to add new flag parameters without changing the hash. Define the new flag such that the previous behaviour is happening when this field is set to false.
- All `non_hashable_fields` fields are excluded from the parameter hash calculation. These fields are set when `dcase_util.containers.AppParameterContainer` is constructed, and they usually are fields used to print various values to the console. These fields do not change the system output to be saved onto disk, and hence they are excluded from hash.


## Extending the baseline

Easiest way to extend the baseline system is by modifying system parameters. To do so one needs to create a parameter file with a custom parameter set, and run system with this parameter file.

**Example 1**

In this example, one creates MLP based system. Data processing chain is replaced with a chain which calculated mean over 500 feature vectors. Learner is replaced with a new model definition. Parameter file `extra.yaml`: 
        
    active_set: minimal-mlp
    sets:
      - set_id: minimal-mlp
        description: Minimal MLP system
        data_processing_chain:
          method: mean_aggregation_chain
        data_processing_chain_method_parameters:
          mean_aggregation_chain:
            chain:
              - processor_name: dcase_util.processors.FeatureReadingProcessor
              - processor_name: dcase_util.processors.NormalizationProcessor
                init_parameters:
                  enable: true
              - processor_name: dcase_util.processors.AggregationProcessor
                init_parameters:
                  aggregation_recipe:
                    - mean
                  win_length_frames: 500
                  hop_length_frames: 500
              - processor_name: dcase_util.processors.DataShapingProcessor
                init_parameters:
                  axis_list:
                    - time_axis
                    - data_axis
        learner:
          method: mlp_mini
        learner_method_parameters:
          mlp_mini:
            random_seed: 0
            keras_profile: deterministic
            backend: tensorflow
            validation_set:
              validation_amount: 0.20
              balancing_mode: identifier_two_level_hierarchy
              seed: 0
            data:
              data_format: channels_last
              target_format: same
            model:
              config:
                - class_name: Dense
                  config:
                    units: 50
                    kernel_initializer: uniform
                    activation: relu
                    input_shape:
                      - FEATURE_VECTOR_LENGTH
                - class_name: Dropout
                  config:
                    rate: 0.2
                - class_name: Dense
                  config:
                    units: 50
                    kernel_initializer: uniform
                    activation: relu
                - class_name: Dropout
                  config:
                    rate: 0.2
                - class_name: Dense
                  config:
                    units: CLASS_COUNT
                    kernel_initializer: uniform
                    activation: softmax
            compile:
              loss: categorical_crossentropy
              metrics:
                - categorical_accuracy
            optimizer:
              class_name: Adam
            fit:
              epochs: 50
              batch_size: 64
              shuffle: true
            callbacks:
              StasherCallback:
                monitor: val_categorical_accuracy
                initial_delay: 25

Command to run the system:

    python task1a.py -p extra.yaml

**Example 2**

In this example, one slightly modifies the baseline system to have smaller network. Learner is replaced with modified model definition. Since `cnn` learner method is overloaded, only a subset of the parameters needs to be defined. However, the model config (network definition) has to be redefined fully as list parameters cannot be overloaded partly. Parameter file `extra.yaml`: 
        
    active_set: baseline-minified
    sets:
      - set_id: baseline-minified
        description: Minified DCASE2018 baseline
        learner_method_parameters:
          cnn:
            model:
              constants:
                CONVOLUTION_KERNEL_SIZE: 3            
        
              config:
                - class_name: Conv2D
                  config:
                    input_shape:
                      - FEATURE_VECTOR_LENGTH   # data_axis
                      - INPUT_SEQUENCE_LENGTH   # time_axis
                      - 1                       # sequence_axis
                    filters: 8
                    kernel_size: CONVOLUTION_KERNEL_SIZE
                    padding: CONVOLUTION_BORDER_MODE
                    kernel_initializer: CONVOLUTION_INIT
                    data_format: DATA_FORMAT
                - class_name: Activation
                  config:
                    activation: CONVOLUTION_ACTIVATION
                - class_name: MaxPooling2D
                  config:
                    pool_size:
                      - 5
                      - 5
                    data_format: DATA_FORMAT
                - class_name: Conv2D
                  config:
                    filters: 16
                    kernel_size: CONVOLUTION_KERNEL_SIZE
                    padding: CONVOLUTION_BORDER_MODE
                    kernel_initializer: CONVOLUTION_INIT
                    data_format: DATA_FORMAT
                - class_name: Activation
                  config:
                    activation: CONVOLUTION_ACTIVATION
                - class_name: MaxPooling2D
                  config:
                    pool_size:
                      - 4
                      - 100
                    data_format: DATA_FORMAT
                - class_name: Flatten      
                - class_name: Dense
                  config:
                    units: 100
                    kernel_initializer: uniform
                    activation: relu    
                - class_name: Dense
                  config:
                    units: CLASS_COUNT
                    kernel_initializer: uniform
                    activation: softmax                        
            fit:
                epochs: 100
                                  
Command to run the system:

    python task1a.py -p extra.yaml


**Example 3**

In this example, multiple different setups are run in a sequence. Parameter file `extra.yaml`: 
        
    active_set: baseline-kernel3
    sets:
      - set_id: baseline-kernel3
        description: DCASE2018 baseline with kernel 3
        learner_method_parameters:
            cnn:
              model:
                constants:
                  CONVOLUTION_KERNEL_SIZE: 3
              fit:
                epochs: 100                    
      - set_id: baseline-kernel5
        description: DCASE2018 baseline with kernel 5
        learner_method_parameters:
            cnn:
              model:
                constants:
                  CONVOLUTION_KERNEL_SIZE: 5
              fit:
                epochs: 100
                
Command to run the system:

    python task1a.py -p extra.yaml -s baseline-kernel3,baseline-kernel5

To see results:
    
    python task1.py --show_results

Code
====

The code is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox, see [manual for tutorials](https://dcase-repo.github.io/dcase_util/index.html). The machine learning part of the code in built on [Keras (v2.1.5)](https://keras.io/), using [TensorFlow (v1.7.0)](https://www.tensorflow.org/) as backend.

### File structure

      .
      ├── task1a.py             # Baseline system for subtask A
      ├── task1a.yaml           # Configuration file for task1a.py
      ├── task1b.py             # Baseline system for subtask B
      ├── task1b.yaml           # Configuration file for task1b.py
      ├── utils.py              # Common functions shared between task1a.py and task1b.py
      ├── README.md             # This file
      └── requirements.txt      # External module dependencies

Changelog
=========

#### 1.0.0 / 2018-04-30

* First public release

License
=======

This software is released under the terms of the [MIT License](https://github.com/DCASE-REPO/dcase2018_baseline/blob/master/LICENSE).
