#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DCASE 2018
# Task 1B: Acoustic Scene Classification with mismatched recording devices
# Baseline system
# ---------------------------------------------
# Author: Toni Heittola ( toni.heittola@tut.fi ), Tampere University of Technology / Audio Research Group
# License: MIT

import dcase_util
import sys
import numpy
import os
import sed_eval

from task1a import do_feature_extraction, do_feature_normalization, do_testing
from utils import *

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


def main(argv):
    # Read application default parameter file
    parameters = dcase_util.containers.DictContainer().load(
        filename='task1b.yaml'
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
        application_title='Task 1B: Acoustic Scene Classification with mismatched recording devices',
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
        logging_file=os.path.join(param.get_path('path.log'), 'task1b.log')
    )

    # Get logging interface
    log = dcase_util.ui.ui.FancyLogger()

    # Log title
    log.title('DCASE2018 / Task1B -- Acoustic Scene Classification with mismatched recording devices')
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
                    do_evaluation_task1b_eval(
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

                    do_evaluation_task1b_leaderboard(
                        db=db_eval,
                        folds=active_folds,
                        param=param,
                        log=log
                    )

                    timer.stop()

                    log.foot(
                        time=timer.elapsed(),
                    )

    return 0


def do_learning(db, folds, param, log, overwrite=False):
    """Learning stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    folds : list of int
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    overwrite : bool
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

            # Setup keras, run only once
            dcase_util.keras.setup_keras(
                seed=param.get_path('learner.parameters.random_seed'),
                profile=param.get_path('learner.parameters.keras_profile'),
                backend=param.get_path('learner.parameters.backend', 'tensorflow'),
                print_indent=2
            )
            import keras

            # Initialize model to None
            keras_model = None

            for pass_id, pass_param in enumerate(param.get_path('learner.parameters.learning_passes', [])):
                # Get learning pass model filename
                pass_model_filename = os.path.join(
                    param.get_path('path.application.learner'),
                    'model_fold_{fold}_pass_{pass_id}.h5'.format(fold=fold, pass_id=pass_id)
                )
                pass_param_container = dcase_util.containers.DictContainer(pass_param)
                if not os.path.isfile(pass_model_filename) or overwrite:
                    log.line('Learning pass [{label}]'.format(label=pass_param['label']), indent=2)

                    # Get fit parameters
                    if pass_param_container.get_path('fit.epochs'):
                        epochs = pass_param_container.get_path('fit.epochs')
                    else:
                        epochs = param.get_path('learner.parameters.fit.epochs')

                    if pass_param_container.get_path('fit.batch_size'):
                        batch_size = pass_param_container.get_path('fit.batch_size')
                    else:
                        batch_size = param.get_path('learner.parameters.fit.batch_size')

                    if pass_param_container.get_path('fit.shuffle'):
                        shuffle = pass_param_container.get_path('fit.shuffle')
                    else:
                        shuffle = param.get_path('learner.parameters.fit.shuffle')

                    # Get all training meta
                    training_meta = db.train(fold=fold)

                    # Filter training meta to contain only asked source devices
                    active_source_labels = pass_param.get('active_source_labels', [])
                    training_meta = training_meta.filter(
                        source_label_list=active_source_labels
                    )

                    if param.get_path('learner.parameters.validation_set') and param.get_path(
                            'learner.parameters.validation_set.enable', True):
                        # Get validation files
                        training_files, validation_files = db.validation_split(
                            training_meta=training_meta,
                            split_type='balanced',
                            validation_amount=param.get_path('learner.parameters.validation_set.validation_amount'),
                            balancing_mode=param.get_path('learner.parameters.validation_set.balancing_mode'),
                            seed=param.get_path('learner.parameters.validation_set.seed', 0),
                            verbose=True
                        )

                    else:
                        # No validation set used
                        training_files = training_meta.unique_files
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
                        target_format = param.get_path('learner.parameters.data.target_format',
                                                       'single_target_per_sequence')
                        data_format = param.get_path('learner.parameters.data.data_format', 'channels_first')

                        log.line('Collecting training data', indent=2)
                        X_train, Y_train, data_size = dcase_util.keras.data_collector(
                            item_list=item_list_train,
                            data_processing_chain=data_processing_chain,
                            meta_processing_chain=meta_processing_chain,
                            target_format=target_format,
                            channel_dimension=data_format,
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
                                target_format=target_format,
                                channel_dimension=data_format,
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

                    if keras_model is None:
                        # Create sequential model, otherwise reuse existing model.
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
                            epochs=epochs,
                            metric=param.get_path('learner.parameters.compile.metrics')[0],
                            loss=param.get_path('learner.parameters.compile.loss'),
                            output_type='logging'
                        )
                    ]

                    if param.get_path('learner.parameters.callbacks.StopperCallback'):
                        # StopperCallback
                        callback_list.append(
                            dcase_util.keras.StopperCallback(
                                epochs=epochs,
                                **param.get_path('learner.parameters.callbacks.StopperCallback', {})
                            )
                        )

                    if param.get_path('learner.parameters.callbacks.ProgressPlotterCallback'):
                        # ProgressPlotterCallback
                        callback_list.append(
                            dcase_util.keras.ProgressPlotterCallback(
                                epochs=epochs,
                                **param.get_path('learner.parameters.callbacks.ProgressPlotterCallback', {})
                            )
                        )

                    if param.get_path('learner.parameters.callbacks.StasherCallback'):
                        # StasherCallback
                        callback_list.append(
                            dcase_util.keras.StasherCallback(
                                epochs=epochs,
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
                            epochs=epochs,
                            shuffle=shuffle
                        )

                    else:
                        keras_model.fit(
                            x=X_train,
                            y=Y_train,
                            validation_data=validation_data,
                            callbacks=callback_list,
                            verbose=0,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=shuffle
                        )

                    # Fetch the best performing model
                    for callback in callback_list:
                        if isinstance(callback, dcase_util.keras.StasherCallback):
                            callback.log()
                            best_weights = callback.get_best()['weights']

                            if best_weights:
                                keras_model.set_weights(best_weights)

                            break

                    # Save model
                    keras_model.save(pass_model_filename)

                else:
                    keras_model = keras.models.load_model(pass_model_filename)

            keras_model.save(fold_model_filename)

            processed_files.append(fold_model_filename)

    return processed_files


def do_evaluation(db, folds, param, log, application_mode='default'):
    """Evaluation stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    folds : list of int
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

    devices = ['a', 'b', 'c']
    device_evaluated = [False, False, False, False]
    class_wise_results = numpy.zeros((len(devices), len(db.scene_labels())))
    device_results = {}
    device_overall_results = {}
    device_class_wise_results = {}

    fold = folds[0]

    fold_results_filename = os.path.join(
        param.get_path('path.application.recognizer'),
        'res_fold_{fold}.txt'.format(fold=fold)
    )

    reference_scene_list_ = db.eval(fold=fold)
    for item_id, item in enumerate(reference_scene_list_):
        item['filename'] = os.path.split(item.filename)[-1]
        item['file'] = item.filename

    estimated_scene_list_ = dcase_util.containers.MetaDataContainer().load(
        filename=fold_results_filename,
        file_format=dcase_util.utils.FileFormat.CSV,
        fields=['filename', 'scene_label'],
        csv_header=False,
        delimiter='\t'
    )
    for item_id, item in enumerate(estimated_scene_list_):
        item['source_label'] = os.path.splitext(os.path.split(item.filename)[-1])[0].split('-')[-1]
        item['filename'] = os.path.split(item.filename)[-1]
        item['file'] = item.filename

    for device_id, device in enumerate(devices):
        reference_scene_list = reference_scene_list_.filter(source_label=device)

        if reference_scene_list:
            device_evaluated[device_id] = True

        estimated_scene_list = estimated_scene_list_.filter(source_label=device)

        evaluator = sed_eval.scene.SceneClassificationMetrics(
            scene_labels=db.scene_labels()
        )

        evaluator.evaluate(
            reference_scene_list=reference_scene_list,
            estimated_scene_list=estimated_scene_list
        )

        results = dcase_util.containers.DictContainer(evaluator.results())
        all_results.append(results)

        device_results[device] = results
        current_class_wise_results = []
        for scene_label_id, scene_label in enumerate(db.scene_labels()):
            class_wise_results[device_id, scene_label_id] = results.get_path(
                ['class_wise', scene_label, 'accuracy', 'accuracy']
            )
            current_class_wise_results.append(
                results.get_path(['class_wise', scene_label, 'accuracy', 'accuracy'])
            )

        overall.append(
            results.get_path('class_wise_average.accuracy.accuracy')
        )

        device_overall_results[device] = results.get_path('class_wise_average.accuracy.accuracy')
        device_class_wise_results[device] = current_class_wise_results

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
            'overall_accuracy': numpy.mean(class_wise_results[1:3, :]),
            'all_results': all_results,
            'parameters': current_param,
            'device_results': device_results,
            'device_overall_results': device_overall_results,
            'device_class_wise_results': device_class_wise_results
        }
    ).save(
        filename=os.path.join(param.get_path('path.application.evaluator'), filename)
    )

    log.line()
    log.row_reset()
    log.row(
        '',
        'High-quality',
        'Mobile devices',
        '',
        widths=[20, 15, 30, 17],
        separators=[True, True, True]
    )
    log.row(
        'Scene',
        'Device [A]',
        'Device [B]',
        'Device [C]',
        'Average [B/C]',
        widths=[20, 15, 15, 15, 17],
        types=['str20', 'float1_percentage', 'float1_percentage', 'float1_percentage', 'float1_percentage'],
        separators=[True, True, False, True, False],
        indent=2
    )
    log.row_sep()

    for scene_label_id, scene_label in enumerate(db.scene_labels()):
        log.row(
            scene_label,
            class_wise_results[0, scene_label_id] * 100.0,
            class_wise_results[1, scene_label_id] * 100.0,
            class_wise_results[2, scene_label_id] * 100.0,
            numpy.mean(class_wise_results[1:3, scene_label_id]) * 100.0
        )

    log.row_sep()
    log.row(
        'Average',
        numpy.mean(class_wise_results[0, :]) * 100.0,
        numpy.mean(class_wise_results[1, :]) * 100.0,
        numpy.mean(class_wise_results[2, :]) * 100.0,
        numpy.mean(class_wise_results[1:3, :]) * 100.0
    )
    log.line()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))

    except (ValueError, IOError) as e:
        sys.exit(e)
