DCASE2018 Baseline systems
==========================

This repository contains the baseline systems for [DCASE2018 challenge](http://dcase.community/challenge2018/) tasks. Task specific baseline systems can be found from subdirectories in this repository.

All baseline systems are implemented in Python, and most of them are built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox. The machine learning part of these systems are based on [Keras](https://keras.io/) while using [TensorFlow](https://www.tensorflow.org/) as backend. 

## Systems

### Task 1 - Acoustic scene classification

The goal of acoustic scene classification is to classify a test recording into one of the provided predefined classes that characterizes the environment in which it was recorded. Audio data recorded in different large european cities will provide a new challenging problem by introducing more acoustic variability for each class than the previous editions.

More information about this task can be found in [task description page](http://dcase.community/challenge2018/task-acoustic-scene-classification).

### Task 2 - General-purpose audio tagging of Freesound content with AudioSet labels

The task evaluates systems for general-purpose audio tagging with an increased number of categories and using data with annotations of varying reliability. This poses the challenges of classifying sound events of very diverse nature (including musical instruments, human sounds, domestic sounds, animals, etc.) and leveraging subsets of training data with annotations of different quality levels. The data used are audio samples from Freesound organized by some categories of the AudioSet Ontology. This task will provide insight towards the development of broadly-applicable sound event classifiers that consider an increased and diverse amount of categories. These models can be used, for example, in automatic description of multimedia or acoustic monitoring applications.

More information about this task can be found in [task description page](http://dcase.community/challenge2018/task-general-purpose-audio-tagging).

### Task 3 - Bird audio detection

The task is to design a system that, given a short audio recording, returns a binary decision for the presence/absence of bird sound (bird sound of any kind).

Please note: the Task 3 baseline is included from a separate git repository. It is included here as a "submodule". If you are using git to download the files, you can run `git submodule init && git submodule update`. Otherwise, you can simply download the Task 3 baseline separately from https://github.com/DCASE-REPO/bulbul_bird_detection_dcase2018

More information about this task can be found in [task description page](http://dcase.community/challenge2018/task-bird-audio-detection).

### Task 4 - Large-scale weakly labeled semi-supervised sound event detection in domestic environments

The task evaluates systems for the large-scale detection of sound events using weakly labeled data. The challenge is to explore the possibility to exploit a large amount of unbalanced and unlabelled training data together with a small weakly annotated training set to improve system performance. The data are YouTube video excerpts focusing on domestic context which could be used for example in ambient assisted living applications. The domain was chosen due to the scientific challenges (wide variety of sounds, time-localized events...) and potential industrial applications.

More information about this task can be found in [task description page](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection).

### Task 5 - Monitoring of domestic activities based on multi-channel acoustics

There is a rising interest in smart environments that enhance the quality of live for humans in terms of e.g. safety, security, comfort, and home care. In order to have smart functionality, situational awareness is required, which might be obtained by interpreting a multitude of sensing modalities including acoustics. The latter is already used in vocal assistants such as Google Home, Apple HomePod, and Amazon Echo. While these devices focus on speech, they could be extended to identify domestic activities carried out by humans. In the literature, this recognition of activities based on acoustics is already touched upon. Yet, the acoustic models are typically based on single channel and single location recordings. In this task, it is investigated to which extend multi-channel acoustic recordings are beneficial for the purpose of detecting domestic activities.

More information about this task can be found in [task description page](http://dcase.community/challenge2018/task-monitoring-domestic-activities).

## License

The baseline systems are released under the terms of the [MIT License](https://github.com/DCASE-REPO/dcase2018_baseline/blob/master/LICENSE).
