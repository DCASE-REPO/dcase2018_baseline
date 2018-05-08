# Dataset for DCASE2018 task4: Large-scale weakly labeled semi-supervised sound event detection in domestic environments

This repository contain a python script to retrieve audio files for DCASE2018 from youtube, the list of audio files and the annotations verified by the task organizers. For a detailed description of the task please visit [task 4 offcial page](http://dcase.community//challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection)

# Commits:
    - April, 9, 2018, 16:36 Removing Y7LDMN0g-M_k_370.000_380.000.wav file in unlabel_out_of_domain.csv. This file is empty.

## Download_data.py

### Dependencies

download_data.py needs Python >= 2.7, dcase_util >= 0.1.6, tqdm >= 4.11.2, ffmpeg>= 2.8.11

A simplified installation procedure example is provide below for python 3.6 based Anconda distribution for Linux based system:
1. [install Ananconda](https://www.anaconda.com/download/#linux) (the script was originally developped for python 3.6)
2. run `bash conda_create_environment.sh` in a terminal
3. before each use of the download script don't forget to active the environment (run `source activate dcase2018`)


**Note:** `The download script has been tested with python 2.7 and 3.6, on linux (ubtuntu 16.04) and Mac OS High Sierra (python 3.6 only) but not on Windows.`



### Instructions

* Clone the repository
* Install the dependencies
* Run the script from the dataset root directory

**Note:** `The final dataset is 82Gb, the download/extraction process can take approximately 12 hours. You can stop the script and resume download later on (see below).`

The script will download the files from youtube, extract the 10-second clips, convert them to audio and store them in the corresponding directory (see also [Dataset structure](#dset_struct)). In case of failure/interruption, upon relaunch, the script will first check and skip the files that were previously downloaded.

The script produces  `missing_files[dataset].csv` log files (were `[dataset]` corresponds to the name of a particular set) with a list of audio files that were not downloaded by the script. After completion if some of the files where not downloaded you might want to run the script a second to download missing files. If you are experiencing problems downloading the full dataset please contact the task organizers (see also [task 4 offcial page](http://dcase.community//challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection))

The script can run several jobs in parallel. This parameter can be adjusted by setting the value `N_JOBS`. Note that the script was tested with 4 jobs and might become unstable with a higher number of jobs.


## <a name="dset_struct"></a> Dataset structure
The content of the development set is structured in the following manner:

```
dataset root
│   readme.md				         (instructions to run the script that downloads the audio files and description of the annotations files)
|   download_data.py                  (script to download the files)
│
└───metadata			              (directories containing the annotations files)
|   │
|   └───train			             (annotations for the train sets)
|   |     weak.csv                    (weakly labeled training set list)
│   |     unlabel_in_domain.csv       (unlabeled in domain training set list)
|   |     unlabel_out_of_domain.csv   (unlabeled out-of-domain training set list)
|   |
|   └───test			              (annotations for the test set)
|         test.csv                    (test set list with strong labels)
|    
└───audio					         (directories where the audio files will be downloaded)
    └───train			             (annotations for the train sets)
    |   └───weak                      (weakly labeled training set)
    |   └───unlabel_in_domain         (unlabeled in domain training set)
    |   └───unlabel_out_of_domain     (unlabeled out-of-domain training set)
    |
    └───test			              (test set)       
```

## Annotation format

### Weak annotations
The weak annotations have been verified manually for a small subset of the training set. The weak annotations are provided in a tab separated csv file under the following format:

```
[filename (string)][tab][class_label (strings)]
```

```
Y-BJNMHMZDcU_50.000_60.000.wav	Alarm_bell_ringing;Dog
```

The first column, `Y-BJNMHMZDcU_50.000_60.000.wav`, is the name of the audio file downloaded from Youtube (`Y-BJNMHMZDcU` is Youtube ID of the video from where the 10-second clips was extracted t=50 sec to t=60 sec, correspond to the clip boundaries within the full video) and the last column, `Alarm_bell_ringing;Dog` corresponds to the sound classes present in the clip separated by a semi-colon.

### Strong annotations
Another subset of the development has been annotated manually with strong annotations, to be used as the test set (see also below for a detailed explanation about the development set). The minimum length for an event is 250ms. The minimum duration of the pause between two events from the same class is 150ms. When the silence between two consecutive events from the same class was less than 150ms the events have been merged to a single event. The strong annotations are provided in a tab separated csv file under the following format:

```
[filename (string)][tab][event onset time in seconds (float)][tab][event offset time in seconds (float)][tab][class_label (strings)]
```
For example:

```
YOTsn73eqbfc_10.000_20.000.wav	0.163	0.665	Alarm_bell_ringing
```

The first column, `YOTsn73eqbfc_10.000_20.000.wav`, is the name of the audio file downloaded from Youtube, the second column `0.163` is the onset time in seconds, the third column `0.665` is the offset time in seconds and the last column, `Alarm_bell_ringing` corresponds to the class of the sound event.



## Authors

Nicolas Turpault, Romain Serizel, Hamid Eghbal-Zadeh, Ankit Parag Shah, 2018 -- Present

## License

This software is distributed under the terms of the MIT License  (https://opensource.org/licenses/MIT)
