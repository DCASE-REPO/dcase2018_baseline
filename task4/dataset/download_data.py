# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah, 2018, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################
from __future__ import print_function, absolute_import

import os
from tqdm import tqdm
from dcase_util.containers import AudioContainer
from youtube_dl.utils import ExtractorError, DownloadError
import youtube_dl
import pandas as pd
import glob
from contextlib import closing
from multiprocessing import Pool
import functools
import shutil

def download_file(result_dir, filename):
    """ download a file from youtube given an audioSet filename. (It takes only a part of the file thanks to
    information provided in the filename)

    Parameters
    ----------

    result_dir : str, result directory which will contain the downloaded file

    filename : str, AudioSet filename to download

    Return
    ------

    list : list, Empty list if the file is downloaded, otherwise contains the filename and the error associated

    """

    tmp_filename = ""
    audio_container = AudioContainer()
    query_id = filename[1:12]
    segment_start = filename[13:-4].split('_')[0]
    segment_end = filename[13:-4].split('_')[1]

    # Define download parameters
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'tmp/%(id)s.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'prefer_ffmpeg': True,
        'logger': MyLogger(),
        'audioformat': 'wav'
    }

    try:
        # Download file
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            meta = ydl.extract_info(
                'https://www.youtube.com/watch?v={query_id}'.format(query_id=query_id), download=True)

        audio_formats = [f for f in meta["formats"] if f.get('vcodec') == 'none']

        if audio_formats is []:
            return [filename, "no audio format available"]

        # get the best audio format
        best_audio_format = audio_formats[-1]

        tmp_filename = "tmp/" + query_id + "." + best_audio_format["ext"]

        # Format audio
        audio_container.load(filename=tmp_filename, fs=44100, res_type='kaiser_best',
                             start=float(segment_start), stop=float(segment_end))

        # Save segmented audio
        audio_container.filename = filename
        audio_container.detect_file_format()
        audio_container.save(filename=os.path.join(result_dir, filename))

        #Remove temporary file
        os.remove(tmp_filename)
        return []

    except (KeyboardInterrupt, SystemExit):
        # Remove temporary files and current audio file.
        for fpath in glob.glob("tmp/" + query_id + "*"):
            os.remove(fpath)
        raise

    # youtube-dl error, file often removed
    except (ExtractorError, DownloadError) as e:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return [filename, str(e)]

    # multiprocessing can give this error
    except IndexError as e:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        print(filename)
        print(str(e))
        return [filename, "Index Error"]


def download(csv_file, result_dir, n_jobs=1, chunk_size=10):
    """ download files in parallel from youtube given a csv file listing files to download. It also stores not downloaded
    files with their associated error in "missing_files_[csv_file].csv"

       Parameters
       ----------
       csv_file : str, filename of a csv file which contains a column "filename" listing AudioSet filenames to download

       result_dir : str, result directory which will contain downloaded files

       n_jobs : int, number of download to execute in parallel

       chunk_size : int, number of files to download before updating the progress bar. Bigger it is, faster it goes
       because data is filled in memory but progress bar only updates after a chunk is finished.

       Return
       ------

       missing_files : pandas.DataFrame, files not downloaded whith associated error.

       """
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # read metadata file and get only one filename once
    df = pd.read_csv(csv_file, header=0, sep='\t')
    filenames = df["filename"].drop_duplicates()
    # Remove already existing files in folder
    existing_files = [os.path.basename(fpath) for fpath in glob.glob(os.path.join(result_dir, "*"))]
    filenames = filenames[~filenames.isin(existing_files)]

    p = None
    non_existing_files = []
    try:
        if n_jobs == 1:
            for filename in tqdm(filenames):
                non_existing_files.append(download_file(result_dir, filename))
        # multiprocessing
        else:
            with closing(Pool(n_jobs)) as p:
                # Put result_dir as a constant variable with result_dir in download_file
                download_file_alias = functools.partial(download_file, result_dir)

                for val in tqdm(p.imap_unordered(download_file_alias, filenames, chunk_size), total=len(filenames)):
                    non_existing_files.append(val)

        # Store files which gave error
        missing_files = pd.DataFrame(non_existing_files).dropna()
        if not missing_files.empty:
            missing_files.columns = ["filename", "error"]
            missing_files.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              "missing_files_" + csv_file.split('/')[-1]),
                                 index=False, sep="\t")

    except KeyboardInterrupt:
        if p is not None:
            p.terminate()
        raise KeyboardInterrupt

    shutil.rmtree("tmp/")

    return missing_files


# Needed to not print warning which cause breaks in the progress bar.
class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


if __name__ == "__main__":
    from dcase_util.ui.ui import FancyLogger
    from dcase_util.utils import setup_logging

    setup_logging(logging_file='download_data.log')
    log = FancyLogger()

    log.title("Download_data")
    log.info("Once database is downloaded, do not forget to check your missing_files")

    # Modify it with the number of process you want, but be careful, youtube can block you if you put too many.
    N_JOBS = 3

    # Only useful when multiprocessing,
    # if chunk_size is high, download is faster. Be careful, progress bar only update after each chunk.
    CHUNK_SIZE = 10

    log.line("Test data")
    test = os.path.join("metadata", "test", "test.csv")
    result_dir = os.path.join("audio", "test")
    download(test, result_dir, n_jobs=N_JOBS, chunk_size=CHUNK_SIZE)

    log.line("Train, weak data")
    train_weak = os.path.join("metadata", "train", "weak.csv")
    result_dir = os.path.join("audio", "train", "weak")
    download(train_weak, result_dir, n_jobs=N_JOBS, chunk_size=CHUNK_SIZE)

    log.line("Train, unlabel in domain data")
    train_unlabel_in_domain = os.path.join("metadata", "train", "unlabel_in_domain.csv")
    result_dir = os.path.join("audio", "train", "unlabel_in_domain")
    download(train_unlabel_in_domain, result_dir, n_jobs=N_JOBS, chunk_size=CHUNK_SIZE)

    log.line("Train, unlabel out of domain data")
    train_unlabel_out_of_domain = os.path.join("metadata", "train", "unlabel_out_of_domain.csv")
    result_dir = os.path.join("audio", "train", "unlabel_out_of_domain")
    download(train_unlabel_out_of_domain, result_dir, n_jobs=N_JOBS, chunk_size=CHUNK_SIZE)

    log.foot()
