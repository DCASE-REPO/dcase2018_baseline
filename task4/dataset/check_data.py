# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah, 2018, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

import pandas as pd
import glob
import os
import argparse
from dcase_util.ui.ui import FancyLogger
from dcase_util.utils import setup_logging

setup_logging(logging_file='check_data.log')
log = FancyLogger()


# This function is not used in the baseline but can be used to check if your audio folders correspond to your metadata
def check_audio_vs_meta(csv_file, audio_dir, write=False):
    """ Check AudioSet filenames contained in csv_file are all present in the resulting audio directory

        Parameters
        ----------

        csv_file : str, filename of a csv file which contains a column "filename" listing AudioSet filenames downloaded

        audio_dir : str, audio directory which contains downloaded files

        write : bool, Write the missing files into a csv file or not.

        Return
        ------

    """
    # read metadata file and get only one filename once
    df = pd.read_csv(csv_file, header=0, sep='\t')
    filenames = df["filename"].drop_duplicates()

    # Remove already existing files in folder
    existing_files = [os.path.basename(fpath) for fpath in glob.glob(os.path.join(audio_dir, "*"))]
    missing_filenames = filenames[~filenames.isin(existing_files)]
    existing_files = pd.Series(existing_files)
    exceed_filenames = existing_files[~existing_files.isin(filenames)]

    log.line("number of missing filenames : " + str(len(missing_filenames)))
    log.line("number of files in audio but not in metadata: " + str(len(exceed_filenames)))

    if write:
        log.line("writing missing and exceed files")
        if not missing_filenames.empty:
            missing_filenames.to_csv("missing_files_" + csv_file.split('/')[-1], index=None)
        if not exceed_filenames.empty:
            exceed_filenames.to_csv("exceed_files_" + csv_file.split('/')[-1], index=None)

    log.foot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()

    test = os.path.join("metadata", "test", "test.csv")
    audio_dir = os.path.join("audio", "test")
    log.line(test)
    check_audio_vs_meta(test, audio_dir, args)

    train_weak = os.path.join("metadata", "train", "weak.csv")
    audio_dir = os.path.join("audio", "train", "weak")
    log.line(train_weak)
    check_audio_vs_meta(train_weak, audio_dir, args)

    train_unlabel_in_domain = os.path.join("metadata", "train", "unlabel_in_domain.csv")
    audio_dir = os.path.join("audio", "train", "unlabel_in_domain")
    log.line(train_unlabel_in_domain)
    check_audio_vs_meta(train_unlabel_in_domain, audio_dir, args)

    train_unlabel_out_of_domain = os.path.join("metadata", "train", "unlabel_out_of_domain.csv")
    audio_dir = os.path.join("audio", "train", "unlabel_out_of_domain")
    log.line(train_unlabel_out_of_domain)
    check_audio_vs_meta(train_unlabel_out_of_domain, audio_dir, args)