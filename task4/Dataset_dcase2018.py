# !/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# This code is an adaptation from Toni Heittola's code (from dcase 2017 task4) [dcase_util](https://dcase-repo.github.io/dcase_util/generated/dcase_util.datasets.DCASE2017_Task4tagging_DevelopmentSet.html)
# Copyright Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah, 2018, v1.0
# This software is distributed under the terms of the License MIT
##########################################################################
from __future__ import print_function, absolute_import

import os
import pandas

from dcase_util.datasets import AudioTaggingDataset
from dcase_util.containers import MetaDataContainer, MetaDataItem, ListDictContainer, AudioContainer, DictContainer
from dcase_util.utils import Path
from dcase_util.tools import FancyLogger
from dataset.download_data import download


class DCASE2018_Task4_DevelopmentSet(AudioTaggingDataset):
    """DCASE 2018 Large-scale weakly labeled semi-supervised sound event detection in domestic environments

    """

    def __init__(self,
                 storage_name='DCASE2018-task4-development',
                 data_path=None,
                 local_path=None,
                 included_content_types=None,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------

        storage_name : str
            Name to be used when storing dataset on disk

        data_path : str
            Root path where the dataset is stored. If None, os.path.join(tempfile.gettempdir(), 'dcase_util_datasets')
            is used.

        local_path : str
            Direct storage path setup for the dataset. If None, data_path and storage_name are used to create one.

        """

        kwargs['included_content_types'] = included_content_types
        kwargs['data_path'] = data_path
        kwargs['storage_name'] = storage_name
        kwargs['local_path'] = local_path
        kwargs['dataset_group'] = 'event'
        kwargs['dataset_meta'] = {
            'authors': 'Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah',
            'title': 'Task 4 Large-scale weakly labeled semi-supervised sound event detection in domestic environments',
            'url': 'https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task4/',
            'audio_source': 'Field recording',
            'audio_type': 'Natural',
            'recording_device_model': None,
            'microphone_model': None,
            'licence': 'MIT'
        }
        kwargs['crossvalidation_folds'] = 2
        kwargs['default_audio_extension'] = 'wav'
        kwargs['package_list'] = [
            {
                'content_type': 'meta',
                'filename': 'dataset/metadata/train/weak.csv'
            },
            {
                'content_type': 'meta',
                'filename': 'dataset/metadata/train/unlabel_in_domain.csv'
            },
            {
                'content_type': 'meta',
                'filename': 'dataset/metadata/test/test.csv'
            },
            {
                'content_type': 'meta',
                'filename': 'dataset/metadata/train/unlabel_out_of_domain.csv',
            },
        ]
        super(DCASE2018_Task4_DevelopmentSet, self).__init__(**kwargs)

    def extract_packages(self):
        """Extract the dataset packages

        Raises
        ------
        IOError
            Local package was not found.

        Returns
        -------
        self

        """
        # Make sure evaluation_setup directory exists
        Path().makedirs(path=os.path.join(self.local_path, self.evaluation_setup_folder))

        log = FancyLogger()

        item_access_log_filename = os.path.join(self.local_path, 'item_access_error.log.csv')
        if 'audio' in self.included_content_types or self.included_content_types == ['all']: # mean process audio
            log.title("Download_data")
            log.info("Once database is downloaded, do not forget to check your missing_files")

            non_existing_videos = pandas.DataFrame(columns=["filename", "error"])

            log.line("check files exist or download data")
            # Collect file ids
            for package in self.package_list:
                if package.get('content_type') == "meta":
                    base_filepath = os.path.splitext(package.get('filename').split('/')[-1])[0]
                    if 'train' in package.get('filename'):
                        result_audio_directory = os.path.join(self.local_path, 'dataset/audio/train', base_filepath)
                    else:
                        result_audio_directory = os.path.join(self.local_path, 'dataset/audio/test')

                    missing_files = download(package.get('filename'), result_audio_directory, n_jobs=3)
                    if not missing_files.empty:
                        non_existing_videos = non_existing_videos.append(missing_files, ignore_index=True)

            # Save list of non-accessible videos
            ListDictContainer(non_existing_videos.to_dict(orient="records"), filename=item_access_log_filename).save(
                fields=['filename', 'error']
            )

        # Evaluation setup filenames
        train_filename_fold1 = self.evaluation_setup_filename(
            setup_part='train',
            fold=1,
            file_extension='csv'
        )

        test_filename_fold1 = self.evaluation_setup_filename(
            setup_part='test',
            fold=1,
            file_extension='csv'
        )

        train_filename_fold2 = self.evaluation_setup_filename(
            setup_part='train',
            fold=2,
            file_extension='csv'
        )

        test_filename_fold2 = self.evaluation_setup_filename(
            setup_part='test',
            fold=2,
            file_extension='csv'
        )

        evaluate_filename = self.evaluation_setup_filename(
            setup_part='evaluate',
            fold=2,
            file_extension='csv'
        )

        # Check that evaluation setup exists
        evaluation_setup_exists = True
        if not os.path.isfile(train_filename_fold1) or not os.path.isfile(test_filename_fold1) \
                or not os.path.isfile(train_filename_fold2) or not os.path.isfile(test_filename_fold2) \
                or not os.path.isfile(evaluate_filename) or not self.meta_container.exists():
            evaluation_setup_exists = False

        if not evaluation_setup_exists:
            # Evaluation setup was not found, generate one
            item_access_log_filename = os.path.join(self.local_path, 'item_access_error.log.csv')
            non_existing_videos = ListDictContainer().load(filename=item_access_log_filename,
                                                           delimiter=',').get_field_unique('filename')

            train_meta_weak_fold1 = MetaDataContainer()
            audio_path = 'dataset/audio/train/weak'
            for item in MetaDataContainer().load(os.path.join(self.local_path, 'dataset/metadata/train/'
                                                                               'weak.csv'),
                                                 fields=["filename", "tags"], csv_header=True):
                if item.filename not in non_existing_videos:
                    if not item.filename.endswith(self.default_audio_extension):
                        item.filename = os.path.join(
                            audio_path,
                            os.path.splitext(item.filename)[0] + '.' + self.default_audio_extension
                        )
                    else:
                        item.filename = Path(path=item.filename).modify(path_base=audio_path)

                    # Only collect items which exists if audio present
                    if 'audio' in self.included_content_types or 'all' in self.included_content_types:
                        if os.path.isfile(os.path.join(self.local_path, item.filename)):
                            train_meta_weak_fold1.append(item)
                    else:
                        train_meta_weak_fold1.append(item)

            train_meta_weak_fold1.save(filename=train_filename_fold1, csv_header=True, file_format="CSV")

            test_meta_unlabel_fold1 = MetaDataContainer()
            audio_path = 'dataset/audio/train/unlabel_in_domain'
            for item in MetaDataContainer().load(os.path.join(self.local_path, 'dataset/metadata/train/'
                                                                               'unlabel_in_domain.csv'),
                                                 csv_header=True):

                if item.filename not in non_existing_videos:
                    # If not the right extension, change it
                    if not item.filename.endswith(self.default_audio_extension):
                        item.filename = os.path.join(
                            audio_path,
                            os.path.splitext(item.filename)[0] + '.' + self.default_audio_extension
                        )
                    else:
                        item.filename = Path(path=item.filename).modify(path_base=audio_path)

                    # Only collect items which exists if audio present
                    if 'audio' in self.included_content_types or 'all' in self.included_content_types:
                        if os.path.isfile(os.path.join(self.local_path, item.filename)):
                            test_meta_unlabel_fold1.append(item)
                    else:
                        test_meta_unlabel_fold1.append(item)

            test_meta_unlabel_fold1.save(filename=test_filename_fold1, csv_header=True, file_format="CSV")

            # Fold 2 train is all the data used in fold 1
            train_meta_weak_fold2 = MetaDataContainer()
            train_meta_weak_fold2 += MetaDataContainer().load(train_filename_fold1, csv_header=True,
                                                              file_format="CSV")

            for item in MetaDataContainer().load(test_filename_fold1, csv_header=True, file_format="CSV"):
                item.tags = []
                train_meta_weak_fold2.append(item)

            train_meta_weak_fold2.save(filename=train_filename_fold2, csv_header=True)

            # Evaluate meta is the groundtruth file with test annotations test.csv
            evaluate_meta = MetaDataContainer()
            audio_path = 'dataset/audio/test'
            for item in MetaDataContainer().load(os.path.join(self.local_path, 'dataset/metadata/test/test.csv'),
                                                 csv_header=True):

                if item.filename not in non_existing_videos:
                    if not item.filename.endswith(self.default_audio_extension):
                        item.filename = os.path.join(audio_path,
                                                     os.path.splitext(item.filename)[0] +
                                                     '.' + self.default_audio_extension)
                    else:
                        item.filename = Path(path=item.filename).modify(path_base=audio_path)

                    # Only collect items which exists
                    if 'audio' in self.included_content_types or 'all' in self.included_content_types:
                        if os.path.isfile(os.path.join(self.local_path, item.filename)):
                            evaluate_meta.append(item)
                    else:
                        evaluate_meta.append(item)

            evaluate_meta.save(filename=evaluate_filename, csv_header=True, file_format="CSV")

            # Test meta is filenames of evaluation, labels will be predicted
            test_meta_strong_fold2 = MetaDataContainer()
            for filename in evaluate_meta.unique_files:
                test_meta_strong_fold2.append(MetaDataItem({'filename': filename}))

            test_meta_strong_fold2.save(filename=test_filename_fold2, csv_header=True, file_format="CSV")

            # meta_data is the default meta container containing all files of the dataset
            meta_data = MetaDataContainer()
            meta_data += MetaDataContainer().load(train_filename_fold1, csv_header=True, file_format="CSV")

            meta_data += MetaDataContainer().load(test_filename_fold1, csv_header=True, file_format="CSV")

            meta_data += MetaDataContainer().load(test_filename_fold2, csv_header=True, file_format="CSV")
            # Save meta
            meta_data.save(filename=self.meta_file)

        log.foot()

        return self

    def load_crossvalidation_data(self):
        """Load cross-validation into the container.

        Returns
        -------
        self

        """

        # Reset cross validation data and insert 'all_data'
        self.crossvalidation_data = DictContainer({
            'train': {
                'all_data': self.meta_container
            },
            'test': {
                'all_data': self.meta_container
            },
            'evaluate': {
                'all_data': self.meta_container
            },
        })

        for crossvalidation_set in list(self.crossvalidation_data.keys()):
            for item in self.crossvalidation_data[crossvalidation_set]['all_data']:
                self.process_meta_item(item=item)

        # Load cross validation folds
        for fold in self.folds():
            # Initialize data
            self.crossvalidation_data['train'][fold] = MetaDataContainer()
            self.crossvalidation_data['test'][fold] = MetaDataContainer()
            self.crossvalidation_data['evaluate'][fold] = MetaDataContainer()

            # Get filenames
            train_filename = self.evaluation_setup_filename(
                setup_part='train',
                fold=fold,
                file_extension="csv"
            )

            test_filename = self.evaluation_setup_filename(
                setup_part='test',
                fold=fold,
                file_extension="csv"
            )

            evaluate_filename = self.evaluation_setup_filename(
                setup_part='evaluate',
                fold=fold,
                file_extension="csv"
            )

            if os.path.isfile(train_filename):
                # Training data for fold exists, load and process it
                self.crossvalidation_data['train'][fold] += MetaDataContainer(filename=train_filename).load()

            if os.path.isfile(test_filename):
                # Testing data for fold exists, load and process it
                self.crossvalidation_data['test'][fold] += MetaDataContainer(filename=test_filename).load()

            if os.path.isfile(evaluate_filename):
                # Evaluation data for fold exists, load and process it
                self.crossvalidation_data['evaluate'][fold] += MetaDataContainer(filename=evaluate_filename).load()

            # Process items
            for item in self.crossvalidation_data['train'][fold]:
                self.process_meta_item(item=item)

            for item in self.crossvalidation_data['test'][fold]:
                self.process_meta_item(item=item)

            for item in self.crossvalidation_data['evaluate'][fold]:
                self.process_meta_item(item=item)

        return self


if __name__ == '__main__':
    dataset = DCASE2018_Task4_DevelopmentSet(storage_name='DCASE2018-task4-development',
                                             included_content_types=None,
                                             local_path=""
                                             )

    dataset.initialize()
    print(dataset)
