#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DCASE 2018
# Task 1A: Acoustic Scene Classification
# Baseline system
# ---------------------------------------------
# Author: Toni Heittola ( toni.heittola@tut.fi ), Tampere University of Technology / Audio Research Group
# License: MIT

import dcase_util
import sys
import numpy
import os
import sed_eval
from utils import *

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


def main(argv):
    # Read application default parameter file
    parameters = dcase_util.containers.DictContainer().load(
        filename='task1a.yaml'
    )

    # Initialize application parameters
    param = dcase_util.containers.DCASEAppParameterContainer(
        parameters,
        path_structure={
            'FEATURE_EXTRACTOR': ['FEATURE_EXTRACTOR'],
            'FEATURE_NORMALIZER': ['FEATURE_EXTRACTOR'],
            'LEARNER': ['DATA_PROCESSING_CHAIN', 'LEARNER'],
            'RECOGNIZER': ['DATA_PROCESSING_CHAIN', 'LEARNER', 'RECOGNIZER'],
        }
    )

    # Handle application arguments
    args = handle_application_arguments(
        param=param,
        application_title='Task 1A: Acoustic Scene Classification',
        version=__version__
    )

    # Process parameters, this is done only after application argument handling in case
    # parameters where injected from command line.
    param.process()

    if args.dataset_path:
        # Download only dataset if requested

        # Make sure given path exists
        dcase_util.utils.Path().create(
            paths=args.dataset_path
        )
        # Get dataset and initialize
        dcase_util.datasets.dataset_factory(
            dataset_class_name=param.get_path('dataset.parameters.dataset'),
            data_path=args.dataset_path,
        ).initialize().log()
        sys.exit(0)

    if args.parameter_set:
        # Check parameter set ids given as program arguments
        parameters_sets = args.parameter_set.split(',')

        # Check parameter_sets
        for set_id in parameters_sets:
            if not param.set_id_exists(set_id=set_id):
                raise ValueError('Parameter set id [{set_id}] not found.'.format(set_id=set_id))

    else:
        parameters_sets = [param.active_set()]

    # Get application mode
    if args.mode:
        application_mode = args.mode

    else:
        application_mode = 'dev'

    # Get overwrite flag
    overwrite = param.get_path('general.overwrite')

    # Make sure all system paths exists
    dcase_util.utils.Path().create(
        paths=list(param['path'].values())
    )

    # Setup logging
    dcase_util.utils.setup_logging(
        logging_file=os.path.join(param.get_path('path.log'), 'task1a.log')
    )

    # Get logging interface
    log = dcase_util.ui.ui.FancyLogger()

    # Log title
    log.title('DCASE2018 / Task1A -- Acoustic scene classification')
    log.line()

    if args.show_results:
        # Show evaluated systems
        show_results(param=param, log=log)
        sys.exit(0)

    if args.show_set_list:
        show_parameter_sets(param=param, log=log)
        sys.exit(0)

    # Create timer instance
    timer = dcase_util.utils.Timer()

    for parameter_set in parameters_sets:
        # Set parameter set
        param['active_set'] = parameter_set
        param.update_parameter_set(parameter_set)

        # Get dataset and initialize
        db = dcase_util.datasets.dataset_factory(
            dataset_class_name=param.get_path('dataset.parameters.dataset'),
            data_path=param.get_path('path.dataset'),
        ).initialize()

        if application_mode == 'eval' or application_mode == 'leaderboard':
            # Application is set to work in 'eval' or 'leaderboard' mode. In these modes, training is done with
            # all data from development dataset, and testing with all data from evaluation dataset.

            # Make sure we are using all data
            active_folds = db.folds(
                mode='full'
            )

        else:
            # Application working in normal mode aka 'dev' mode

            # Get active folds from dataset
            active_folds = db.folds(
                mode=param.get_path('dataset.parameters.evaluation_mode')
            )

            # Get active fold list from parameters
            active_fold_list = param.get_path('general.active_fold_list')

            if active_fold_list and len(set(active_folds).intersection(active_fold_list)) > 0:
                # Active fold list is set and it intersects with active_folds given by dataset class
                active_folds = list(set(active_folds).intersection(active_fold_list))

        # Print some general information
        show_general_information(
            parameter_set=parameter_set,
            active_folds=active_folds,
            param=param,
            db=db,
            log=log
        )

        if param.get_path('flow.feature_extraction'):
            # Feature extraction stage
            log.section_header('Feature Extraction')

            timer.start()

            processed_items = do_feature_extraction(
                db=db,
                param=param,
                log=log,
                overwrite=overwrite
            )

            timer.stop()

            log.foot(
                time=timer.elapsed(),
                item_count=len(processed_items)
            )

        if param.get_path('flow.feature_normalization'):
            # Feature extraction stage
            log.section_header('Feature Normalization')

            timer.start()

            processed_items = do_feature_normalization(
                db=db,
                folds=active_folds,
                param=param,
                log=log,
                overwrite=overwrite
            )
            timer.stop()

            log.foot(
                time=timer.elapsed(),
                item_count=len(processed_items)
            )

        if param.get_path('flow.learning'):
            # Learning stage
            log.section_header('Learning')

            timer.start()

            processed_items = do_learning(
                db=db,
                folds=active_folds,
                param=param,
                log=log,
                overwrite=overwrite
            )

            timer.stop()

            log.foot(
                time=timer.elapsed(),
                item_count=len(processed_items)
            )

        if application_mode == 'dev':
            # System evaluation in 'dev' mode

            if param.get_path('flow.testing'):
                # Testing stage
                log.section_header('Testing')

                timer.start()

                processed_items = do_testing(
                    db=db,
                    folds=active_folds,
                    param=param,
                    log=log,
                    overwrite=overwrite
                )

                timer.stop()

                log.foot(
                    time=timer.elapsed(),
                    item_count=len(processed_items)
                )

                if args.output_file:
                    save_system_output(
                        db=db,
                        folds=active_folds,
                        param=param,
                        log=log,
                        output_file=args.output_file
                    )

            if param.get_path('flow.evaluation'):
                # Evaluation stage
                log.section_header('Evaluation')

                timer.start()

                do_evaluation(
                    db=db,
                    folds=active_folds,
                    param=param,
                    log=log,
                    application_mode=application_mode
                )
                timer.stop()

                log.foot(
                    time=timer.elapsed(),
                )

        elif application_mode == 'eval' or application_mode == 'leaderboard':
            # System evaluation in eval/leaderboard mode
            if application_mode == 'eval':
                # Get set id for eval parameters, test if current set id with eval post fix exists
                eval_parameter_set_id = param.active_set() + '_eval'
                if not param.set_id_exists(eval_parameter_set_id):
                    raise ValueError(
                        'Parameter set id [{set_id}] not found for eval mode.'.format(
                            set_id=eval_parameter_set_id
                        )
                    )

            elif application_mode == 'leaderboard':
                # Get set id for eval parameters, test if current set id with eval post fix exists
                eval_parameter_set_id = param.active_set() + '_leaderboard'
                if not param.set_id_exists(eval_parameter_set_id):
                    raise ValueError(
                        'Parameter set id [{set_id}] not found for leaderboard mode.'.format(
                            set_id=eval_parameter_set_id
                        )
                    )

            # Change active parameter set
            param.update_parameter_set(eval_parameter_set_id)

            # Get eval dataset and initialize
            db_eval = dcase_util.datasets.dataset_factory(
                dataset_class_name=param.get_path('dataset.parameters.dataset'),
                data_path=param.get_path('path.dataset'),
            ).initialize()

            # Get active folds
            active_folds = db_eval.folds(
                mode='full'
            )

            if param.get_path('flow.feature_extraction'):
                # Feature extraction for eval
                log.section_header('Feature Extraction')

                timer.start()

                processed_items = do_feature_extraction(
                    db=db_eval,
                    param=param,
                    log=log,
                    overwrite=overwrite
                )

                timer.stop()

                log.foot(
                    time=timer.elapsed(),
                    item_count=len(processed_items)
                )

            if param.get_path('flow.testing'):
                # Testing stage for eval
                log.section_header('Testing')

                timer.start()

                processed_items = do_testing(
                    db=db_eval,
                    folds=active_folds,
                    param=param,
                    log=log,
                    overwrite=overwrite
                )

                timer.stop()

                log.foot(
                    time=timer.elapsed(),
                    item_count=len(processed_items)
                )

                if args.output_file:
                    save_system_output(
                        db=db_eval,
                        folds=active_folds,
                        param=param,
                        log=log,
                        output_file=args.output_file,
                        mode='leaderboard' if application_mode == 'leaderboard' else 'dcase'
                    )

            if db_eval.reference_data_present and param.get_path('flow.evaluation'):
                if application_mode == 'eval':
                    # Evaluation stage for eval
                    log.section_header('Evaluation')

                    timer.start()

                    do_evaluation(
                        db=db_eval,
                        folds=active_folds,
                        param=param,
                        log=log,
                        application_mode=application_mode
                    )

                    timer.stop()

                    log.foot(
                        time=timer.elapsed(),
                    )

                elif application_mode == 'leaderboard':
                    # Evaluation stage for eval
                    log.section_header('Evaluation')

                    timer.start()

                    do_evaluation_task1a_leaderboard(
                        db=db_eval,
                        folds=active_folds,
                        param=param,
                        log=log,
                        application_mode=application_mode
                    )

                    timer.stop()

                    log.foot(
                        time=timer.elapsed(),
                    )

    return 0


def do_feature_extraction(db, param, log, overwrite=False):
    """Feature extraction stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    list of str

    """

    # Prepare feature extractor
    mel_extractor = dcase_util.features.MelExtractor(
        **param.get_path('feature_extractor.parameters', {})
    )

    # Loop over all audio files in the current dataset and extract acoustic features for each of them.
    processed_files = []
    for item_id, audio_filename in enumerate(db.audio_files):
        # Get filename for feature data from audio filename
        feature_filename = dcase_util.utils.Path(
            path=audio_filename
        ).modify(
            path_base=param.get_path('path.application.feature_extractor'),
            filename_extension='.cpickle'
        )

        if not os.path.isfile(feature_filename) or overwrite:
            log.line(
                data='[{item: >5} / {total}] [{filename}]'.format(
                    item=item_id,
                    total=len(db.audio_files),
                    filename=os.path.split(audio_filename)[1]
                ),
                indent=2
            )

            # Load audio data
            audio = dcase_util.containers.AudioContainer().load(
                filename=audio_filename,
                mono=True,
                fs=param.get_path('feature_extractor.fs')
            )

            # Extract features and store them into FeatureContainer, and save it to the disk
            dcase_util.containers.FeatureContainer(
                data=mel_extractor.extract(audio.data),
                time_resolution=param.get_path('feature_extractor.hop_length_seconds')
            ).save(
                filename=feature_filename
            )
            processed_files.append(feature_filename)

    return processed_files


def do_feature_normalization(db, folds, param, log, overwrite=False):
    """Feature normalization stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    list of str

    """

    # Loop over all active cross-validation folds and calculate mean and std for the training data

    processed_files = []

    for fold in folds:
        log.line(
            data='Fold [{fold}]'.format(fold=fold),
            indent=2
        )

        # Get filename for the normalization factors
        fold_stats_filename = os.path.join(
            param.get_path('path.application.feature_normalizer'),
            'norm_fold_{fold}.cpickle'.format(fold=fold)
        )

        if not os.path.isfile(fold_stats_filename) or overwrite:
            normalizer = dcase_util.data.Normalizer(
                filename=fold_stats_filename
            )

            # Loop through all training data
            for item in db.train(fold=fold):
                # Get feature filename
                feature_filename = dcase_util.utils.Path(
                    path=item.filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                # Load feature matrix
                features = dcase_util.containers.FeatureContainer().load(
                    filename=feature_filename
                )

                # Accumulate statistics
                normalizer.accumulate(
                    data=features
                )

            # Finalize and save
            normalizer.finalize().save()

            processed_files.append(fold_stats_filename)

    return processed_files


def do_learning(db, folds, param, log, overwrite=False):
    """Learning stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    nothing

    """

    # Loop over all cross-validation folds and learn acoustic models

    processed_files = []

    for fold in folds:
        log.line(
            data='Fold [{fold}]'.format(fold=fold),
            indent=2
        )

        # Get model filename
        fold_model_filename = os.path.join(
            param.get_path('path.application.learner'),
            'model_fold_{fold}.h5'.format(fold=fold)
        )

        if not os.path.isfile(fold_model_filename) or overwrite:
            log.line()

            # Get normalization factor filename
            fold_stats_filename = os.path.join(
                param.get_path('path.application.feature_normalizer'),
                'norm_fold_{fold}.cpickle'.format(fold=fold)
            )

            # Create data processing chain for features
            data_processing_chain = dcase_util.processors.ProcessingChain()
            for chain in param.get_path('data_processing_chain.parameters.chain'):
                processor_name = chain.get('processor_name')
                init_parameters = chain.get('init_parameters', {})

                # Inject parameters
                if processor_name == 'dcase_util.processors.NormalizationProcessor':
                    init_parameters['filename'] = fold_stats_filename

                data_processing_chain.push_processor(
                    processor_name=processor_name,
                    init_parameters=init_parameters,
                )

            # Create meta processing chain for reference data
            meta_processing_chain = dcase_util.processors.ProcessingChain()
            for chain in param.get_path('meta_processing_chain.parameters.chain'):
                processor_name = chain.get('processor_name')
                init_parameters = chain.get('init_parameters', {})

                # Inject parameters
                if processor_name == 'dcase_util.processors.OneHotEncodingProcessor':
                    init_parameters['label_list'] = db.scene_labels()

                meta_processing_chain.push_processor(
                    processor_name=processor_name,
                    init_parameters=init_parameters,
                )

            if param.get_path('learner.parameters.validation_set') and param.get_path('learner.parameters.validation_set.enable', True):
                # Get validation files
                training_files, validation_files = db.validation_split(
                    fold=fold,
                    split_type='balanced',
                    validation_amount=param.get_path('learner.parameters.validation_set.validation_amount'),
                    balancing_mode=param.get_path('learner.parameters.validation_set.balancing_mode'),
                    seed=param.get_path('learner.parameters.validation_set.seed', 0),
                    verbose=True
                )

            else:
                # No validation set used
                training_files = db.train(fold=fold).unique_files
                validation_files = dcase_util.containers.MetaDataContainer()

            # Create item_list_train and item_list_validation
            item_list_train = []
            item_list_validation = []
            for item in db.train(fold=fold):
                # Get feature filename
                feature_filename = dcase_util.utils.Path(
                    path=item.filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                item_ = {
                    'data': {
                        'filename': feature_filename
                    },
                    'meta': {
                        'label': item.scene_label
                    }
                }

                if item.filename in validation_files:
                    item_list_validation.append(item_)

                elif item.filename in training_files:
                    item_list_train.append(item_)

            # Setup keras, run only once
            dcase_util.keras.setup_keras(
                seed=param.get_path('learner.parameters.random_seed'),
                profile=param.get_path('learner.parameters.keras_profile'),
                backend=param.get_path('learner.parameters.backend', 'tensorflow'),
                print_indent=2
            )

            if param.get_path('learner.parameters.generator.enable'):
                # Create data generators for training and validation

                # Get generator class, class is inherited from keras.utils.Sequence class.
                KerasDataSequence = dcase_util.keras.get_keras_data_sequence_class()

                # Training data generator
                train_data_sequence = KerasDataSequence(
                    item_list=item_list_train,
                    data_processing_chain=data_processing_chain,
                    meta_processing_chain=meta_processing_chain,
                    batch_size=param.get_path('learner.parameters.fit.batch_size'),
                    data_format=param.get_path('learner.parameters.data.data_format'),
                    target_format=param.get_path('learner.parameters.data.target_format'),
                    **param.get_path('learner.parameters.generator', default={})
                )

                # Show data properties
                train_data_sequence.log()

                if item_list_validation:
                    # Validation data generator
                    validation_data_sequence = KerasDataSequence(
                        item_list=item_list_validation,
                        data_processing_chain=data_processing_chain,
                        meta_processing_chain=meta_processing_chain,
                        batch_size=param.get_path('learner.parameters.fit.batch_size'),
                        data_format=param.get_path('learner.parameters.data.data_format'),
                        target_format=param.get_path('learner.parameters.data.target_format')
                    )

                else:
                    validation_data_sequence = None

                # Get data item size
                data_size = train_data_sequence.data_size

            else:
                # Collect training data and corresponding targets to matrices
                log.line('Collecting training data', indent=2)

                X_train, Y_train, data_size = dcase_util.keras.data_collector(
                    item_list=item_list_train,
                    data_processing_chain=data_processing_chain,
                    meta_processing_chain=meta_processing_chain,
                    target_format=param.get_path('learner.parameters.data.target_format', 'single_target_per_sequence'),
                    channel_dimension=param.get_path('learner.parameters.data.data_format', 'channels_first'),
                    verbose=True,
                    print_indent=4
                )
                log.foot(indent=2)

                if item_list_validation:
                    log.line('Collecting validation data', indent=2)
                    X_validation, Y_validation, data_size = dcase_util.keras.data_collector(
                        item_list=item_list_validation,
                        data_processing_chain=data_processing_chain,
                        meta_processing_chain=meta_processing_chain,
                        target_format=param.get_path('learner.parameters.data.target_format', 'single_target_per_sequence'),
                        channel_dimension=param.get_path('learner.parameters.data.data_format', 'channels_first'),
                        verbose=True,
                        print_indent=4
                    )
                    log.foot(indent=2)

                    validation_data = (X_validation, Y_validation)

                else:
                    validation_data = None

            # Collect constants for the model generation, add class count and feature matrix size
            model_parameter_constants = {
                'CLASS_COUNT': int(db.scene_label_count()),
                'FEATURE_VECTOR_LENGTH': int(data_size['data']),
                'INPUT_SEQUENCE_LENGTH': int(data_size['time']),
            }

            # Read constants from parameters
            model_parameter_constants.update(
                param.get_path('learner.parameters.model.constants', {})
            )

            # Create sequential model
            keras_model = dcase_util.keras.create_sequential_model(
                model_parameter_list=param.get_path('learner.parameters.model.config'),
                constants=model_parameter_constants
            )

            # Create optimizer object
            param.set_path(
                path='learner.parameters.compile.optimizer',
                new_value=dcase_util.keras.create_optimizer(
                    class_name=param.get_path('learner.parameters.optimizer.class_name'),
                    config=param.get_path('learner.parameters.optimizer.config')
                )
            )

            # Compile model
            keras_model.compile(
                **param.get_path('learner.parameters.compile', {})
            )

            # Show model topology
            log.line(
                dcase_util.keras.model_summary_string(keras_model)
            )

            # Create callback list
            callback_list = [
                dcase_util.keras.ProgressLoggerCallback(
                    epochs=param.get_path('learner.parameters.fit.epochs'),
                    metric=param.get_path('learner.parameters.compile.metrics')[0],
                    loss=param.get_path('learner.parameters.compile.loss'),
                    output_type='logging'
                )
            ]

            if param.get_path('learner.parameters.callbacks.StopperCallback'):
                # StopperCallback
                callback_list.append(
                    dcase_util.keras.StopperCallback(
                        epochs=param.get_path('learner.parameters.fit.epochs'),
                        **param.get_path('learner.parameters.callbacks.StopperCallback', {})
                    )
                )

            if param.get_path('learner.parameters.callbacks.ProgressPlotterCallback'):
                # ProgressPlotterCallback
                callback_list.append(
                    dcase_util.keras.ProgressPlotterCallback(
                        epochs=param.get_path('learner.parameters.fit.epochs'),
                        **param.get_path('learner.parameters.callbacks.ProgressPlotterCallback', {})
                    )
                )

            if param.get_path('learner.parameters.callbacks.StasherCallback'):
                # StasherCallback
                callback_list.append(
                    dcase_util.keras.StasherCallback(
                        epochs=param.get_path('learner.parameters.fit.epochs'),
                        **param.get_path('learner.parameters.callbacks.StasherCallback', {})
                    )
                )

            # Train model
            if param.get_path('learner.parameters.generator.enable'):
                keras_model.fit_generator(
                    generator=train_data_sequence,
                    validation_data=validation_data_sequence,
                    callbacks=callback_list,
                    verbose=0,
                    epochs=param.get_path('learner.parameters.fit.epochs'),
                    shuffle=param.get_path('learner.parameters.fit.shuffle')
                )

            else:
                keras_model.fit(
                    x=X_train,
                    y=Y_train,
                    validation_data=validation_data,
                    callbacks=callback_list,
                    verbose=0,
                    epochs=param.get_path('learner.parameters.fit.epochs'),
                    batch_size=param.get_path('learner.parameters.fit.batch_size'),
                    shuffle=param.get_path('learner.parameters.fit.shuffle')
                )

            for callback in callback_list:
                if isinstance(callback, dcase_util.keras.StasherCallback):
                    # Fetch the best performing model
                    callback.log()
                    best_weights = callback.get_best()['weights']

                    if best_weights:
                        keras_model.set_weights(best_weights)

                    break

            # Save model
            keras_model.save(fold_model_filename)

            processed_files.append(fold_model_filename)

    return processed_files


def do_testing(db, folds, param, log, overwrite=False):
    """Testing stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    list

    """

    processed_files = []

    # Loop over all cross-validation folds and test
    for fold in folds:
        log.line(
            data='Fold [{fold}]'.format(fold=fold),
            indent=2
        )

        # Get model filename
        fold_model_filename = os.path.join(
            param.get_path('path.application.learner'),
            'model_fold_{fold}.h5'.format(fold=fold)
        )

        # Initialize model to None, load when first non-tested file encountered.
        keras_model = None

        # Get normalization factor filename
        fold_stats_filename = os.path.join(
            param.get_path('path.application.feature_normalizer'),
            'norm_fold_{fold}.cpickle'.format(fold=fold)
        )

        # Create processing chain for features
        data_processing_chain = dcase_util.processors.ProcessingChain()
        for chain in param.get_path('data_processing_chain.parameters.chain'):
            processor_name = chain.get('processor_name')
            init_parameters = chain.get('init_parameters', {})

            # Inject parameters
            if processor_name == 'dcase_util.processors.NormalizationProcessor':
                init_parameters['filename'] = fold_stats_filename

            data_processing_chain.push_processor(
                processor_name=processor_name,
                init_parameters=init_parameters,
            )

        # Get results filename
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.txt'.format(fold=fold)
        )

        if not os.path.isfile(fold_results_filename) or overwrite:
            # Load model if not yet loaded
            if not keras_model:
                dcase_util.keras.setup_keras(
                    seed=param.get_path('learner.parameters.random_seed'),
                    profile=param.get_path('learner.parameters.keras_profile'),
                    backend=param.get_path('learner.parameters.backend', 'tensorflow'),
                    print_indent=2
                )
                import keras

                keras_model = keras.models.load_model(fold_model_filename)

            # Initialize results container
            res = dcase_util.containers.MetaDataContainer(
                filename=fold_results_filename
            )

            # Loop through all test files from the current cross-validation fold
            for item in db.test(fold=fold):
                # Get feature filename
                feature_filename = dcase_util.utils.Path(
                    path=item.filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                features = data_processing_chain.process(
                    filename=feature_filename
                )
                input_data = features.data

                if len(keras_model.input_shape) == 4:
                    # Add channel
                    if keras_model.get_config()[0]['config']['data_format'] == 'channels_first':
                        input_data = numpy.expand_dims(input_data, 0)

                    elif keras_model.get_config()[0]['config']['data_format'] == 'channels_last':
                        input_data = numpy.expand_dims(input_data, 3)

                # Get network output
                probabilities = keras_model.predict(x=input_data).T

                # Binarization of the network output
                frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                    probabilities=probabilities,
                    binarization_type=param.get_path('recognizer.frame_binarization.type', 'global_threshold'),
                    threshold=param.get_path('recognizer.frame_binarization.threshold', 0.5)
                )

                estimated_scene_label = dcase_util.data.DecisionEncoder(
                    label_list=db.scene_labels()
                ).majority_vote(
                    frame_decisions=frame_decisions
                )

                # Store result into results container
                res.append(
                    {
                        'filename': item.filename,
                        'scene_label': estimated_scene_label
                    }
                )

                processed_files.append(item.filename)

            # Save results container
            res.save()

    return processed_files


def do_evaluation(db, folds, param, log, application_mode='default'):
    """Evaluation stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    application_mode : str
        Application mode
        Default value 'default'

    Returns
    -------
    nothing

    """

    all_results = []
    overall = []

    class_wise_results = numpy.zeros((len(folds), len(db.scene_labels())))
    for fold_id, fold in enumerate(folds):
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.txt'.format(fold=fold)
        )

        reference_scene_list = db.eval(fold=fold)
        for item_id, item in enumerate(reference_scene_list):
            reference_scene_list[item_id]['filename'] = os.path.split(item.filename)[-1]
            reference_scene_list[item_id]['file'] = item.filename

        estimated_scene_list = dcase_util.containers.MetaDataContainer().load(
            filename=fold_results_filename,
            file_format=dcase_util.utils.FileFormat.CSV,
            fields=['filename', 'scene_label'],
            csv_header=False,
            delimiter='\t'
        )

        for item_id, item in enumerate(estimated_scene_list):
            estimated_scene_list[item_id]['filename'] = os.path.split(item.filename)[-1]
            estimated_scene_list[item_id]['file'] = item.filename

        evaluator = sed_eval.scene.SceneClassificationMetrics(
            scene_labels=db.scene_labels()
        )

        evaluator.evaluate(
            reference_scene_list=reference_scene_list,
            estimated_scene_list=estimated_scene_list
        )

        results = dcase_util.containers.DictContainer(evaluator.results())
        all_results.append(results)

        class_wise_accuracy = []
        for scene_label_id, scene_label in enumerate(db.scene_labels()):
            class_wise_accuracy.append(
                results.get_path(
                    ['class_wise', scene_label, 'accuracy', 'accuracy']
                )
            )
            if fold != 'all_data':
                class_wise_results[fold_id, scene_label_id] = results.get_path(
                    ['class_wise', scene_label, 'accuracy', 'accuracy']
                )

            else:
                class_wise_results[0, scene_label_id] = results.get_path(
                    ['class_wise', scene_label, 'accuracy', 'accuracy']
                )

        overall.append(
            results.get_path('class_wise_average.accuracy.accuracy')
        )

    # Get filename
    filename = 'eval_{parameter_hash}_{application_mode}.yaml'.format(
        parameter_hash=param['_hash'],
        application_mode=application_mode
    )

    # Get current parameters
    current_param = dcase_util.containers.AppParameterContainer(param.get_set(param.active_set()))
    current_param._clean_unused_parameters()
    if current_param.get_path('learner.parameters.compile.optimizer'):
        current_param.set_path('learner.parameters.compile.optimizer', None)

    # Save evaluation information
    dcase_util.containers.DictContainer(
        {
            'application_mode': application_mode,
            'set_id': param.active_set(),
            'class_wise_results': class_wise_results,
            'overall_accuracy': numpy.mean(overall),
            'all_results': all_results,
            'parameters': current_param
        }
    ).save(
        filename=os.path.join(param.get_path('path.application.evaluator'), filename)
    )

    log.line()
    log.row_reset()

    column_headers = ['Scene']
    column_widths = [20]
    column_types = ['str20']
    column_separators = [True]
    if len(folds) > 1:
        for fold_id, fold in enumerate(folds):
            column_headers.append('Fold {fold}'.format(fold=fold))
            column_widths.append(15)
            column_types.append('float1_percentage')
            if fold_id == len(folds):
                column_separators.append(False)
            else:
                column_separators.append(True)

        column_headers.append('Average')

    else:
        column_headers.append('Accuracy')

    column_widths.append(25)
    column_types.append('float1_percentage')
    column_separators.append(False)

    log.row(
        *column_headers,
        widths=column_widths,
        types=column_types,
        separators=column_separators,
        indent=2
    )
    log.row_sep()

    for scene_label_id, scene_label in enumerate(db.scene_labels()):

        column_values = [scene_label]
        if len(folds) > 1:
            for fold_id, fold in enumerate(folds):
                column_values.append(
                    class_wise_results[fold_id, scene_label_id]*100.0
                )

        column_values.append(
            numpy.mean(class_wise_results[:, scene_label_id])*100.0
        )

        log.row(*column_values)

    log.row_sep()
    column_values = ['Average']

    if len(folds) > 1:
        for fold_id, fold in enumerate(folds):
            column_values.append(numpy.mean(class_wise_results[fold_id, :])*100.0)

    column_values.append(
        numpy.mean(overall)*100.0
    )

    log.row(
        *column_values,
        types=column_types
    )

    log.line()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))

    except (ValueError, IOError) as e:
        sys.exit(e)
