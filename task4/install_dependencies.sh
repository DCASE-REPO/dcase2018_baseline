#!/bin/bash
conda install scipy h5py pandas
conda install -c conda-forge tqdm
conda install -c conda-forge librosa
conda install -c conda-forge ffmpeg
pip install keras
pip install sed_eval

# tensorflow CPU:
pip install --upgrade tensorflow

# tensorflow gpu, go to: https://www.tensorflow.org/install

pip install dcase_util