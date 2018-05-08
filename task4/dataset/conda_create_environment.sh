#!/bin/bash
conda create -n dcase2018 python=3.6 scipy pandas
source activate dcase2018
conda install -c conda-forge tqdm
conda install -c conda-forge librosa
conda install -c conda-forge ffmpeg
pip install dcase_util
