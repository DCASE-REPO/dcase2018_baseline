# !/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# This code is an adaptation from Toni Heittola's code [task1 baseline dcase 2018](https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task1/)
# Copyright Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah, 2018, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################
import dcase_util
import sys
import numpy
import os
import random
import pickle

import tensorflow as tf
from keras import backend as K
import keras

#from evaluation_measures import get_f_measure_by_class, event_based_evaluation, segment_based_evaluation
from evaluation_measures import get_f_measure_by_class, event_based_evaluation
from Dataset_dcase2018 import DCASE2018_Task4_DevelopmentSet

dcase_util.utils.setup_logging(logging_file='task4.log')
print(keras.__version__)

random.seed(10)
numpy.random.seed(42)

tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)


def main(parameters):
    log = dcase_util.ui.ui.FancyLogger()
    log.title('DCASE2018 / Task4')

    overwirte_preprocessing = False
    overwrite_learning = False
    overwrite_testing = True

    # =====================================================================
    # Parameters
    # =====================================================================
    # Process parameters
    param = dcase_util.containers.DCASEAppParameterContainer(
        parameters,
        path_structure={
            'FEATURE_EXTRACTOR': [
                'DATASET',
                'FEATURE_EXTRACTOR'
            ],
            'FEATURE_NORMALIZER': [
                'DATASET',
                'FEATURE_EXTRACTOR'
            ],
            'LEARNER': [
                'DATASET',
                'FEATURE_EXTRACTOR',
                'FEATURE_NORMALIZER',
                'FEATURE_SEQUENCER',
                'LEARNER'
            ],
            'RECOGNIZER': [
                'DATASET',
                'FEATURE_EXTRACTOR',
                'FEATURE_NORMALIZER',
                'FEATURE_SEQUENCER',
                'LEARNER',
                'RECOGNIZER'
            ],
        }
    ).process()

    # Make sure all system paths exists
    dcase_util.utils.Path().create(
        paths=list(param['path'].values())
    )

    # Initialize
    keras_model_first_pass = None
    keras_model_second_pass = None

    # =====================================================================
    # Dataset
    # =====================================================================
    # Get dataset and initialize it

    db = DCASE2018_Task4_DevelopmentSet(included_content_types=['all'],
                                        local_path="",
                                        data_path=param.get_path('path.dataset'),
                                        audio_paths=[
                                            os.path.join("dataset", "audio", "train", "weak"),
                                            os.path.join("dataset", "audio", "train", "unlabel_in_domain"),
                                            os.path.join("dataset", "audio", "train", "unlabel_out_of_domain"),
                                            os.path.join("dataset", "audio", "test")
                                        ]
                                        ).initialize()

    # Active folds
    folds = db.folds(
        mode=param.get_path('dataset.parameters.evaluation_mode')
    )

    active_fold_list = param.get_path('dataset.parameters.fold_list')
    if active_fold_list:
        folds = list(set(folds).intersection(active_fold_list))

    # =====================================================================
    # Feature extraction stage
    # =====================================================================
    if param.get_path('flow.feature_extraction'):
        log.section_header('Feature Extraction / Train material')

        # Prepare feature extractor
        mel_extractor = dcase_util.features.MelExtractor(
            **param.get_path('feature_extractor.parameters.mel')
        )

        # Loop over all audio files in the dataset and extract features for them.
        # for audio_filename in db.audio_files:
        for audio_filename in db.audio_files:
            # Get filename for feature data from audio filename
            feature_filename = dcase_util.utils.Path(
                path=audio_filename
            ).modify(
                path_base=param.get_path('path.application.feature_extractor'),
                filename_extension='.cpickle'
            )

            if not os.path.isfile(feature_filename) or overwirte_preprocessing:
                log.line(
                    data=os.path.split(audio_filename)[1],
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

        log.foot()

    # =====================================================================
    # Feature normalization stage
    # =====================================================================

    if param.get_path('flow.feature_normalization'):
        log.section_header('Feature Normalization')

        # Get filename for the normalization factors
        features_norm_filename = os.path.join(
            param.get_path('path.application.feature_normalizer'),
            'normalize_values.cpickle'
        )

        if not os.path.isfile(features_norm_filename) or overwirte_preprocessing:
            normalizer = dcase_util.data.Normalizer(
                filename=features_norm_filename
            )

            #  Loop through all training data, two train folds
            for fold in folds:
                for filename in db.train(fold=fold).unique_files:
                    # Get feature filename
                    feature_filename = dcase_util.utils.Path(
                        path=filename
                    ).modify(
                        path_base=param.get_path('path.application.feature_extractor'),
                        filename_extension='.cpickle',
                    )

                    # Load feature matrix
                    features = dcase_util.containers.FeatureContainer().load(
                        filename=feature_filename
                    )

                    # Accumulate statistics
                    normalizer.accumulate(
                        data=features.data
                    )

            # Finalize and save
            normalizer.finalize().save()

        log.foot()

    # Create processing chain for features
    feature_processing_chain = dcase_util.processors.ProcessingChain()
    for chain in param.get_path('feature_processing_chain'):
        processor_name = chain.get('processor_name')
        init_parameters = chain.get('init_parameters', {})

        # Inject parameters
        if processor_name == 'dcase_util.processors.NormalizationProcessor':
            init_parameters['filename'] = features_norm_filename

        if init_parameters.get('enable') is None or init_parameters.get('enable') is True:
            feature_processing_chain.push_processor(
                processor_name=processor_name,
                init_parameters=init_parameters,
            )

    # =====================================================================
    # Learning stage
    # =====================================================================
    if param.get_path('flow.learning'):
        log.section_header('Learning')

        # setup keras parameters
        dcase_util.keras.setup_keras(
            seed=param.get_path('learner.parameters.random_seed'),
            profile=param.get_path('learner.parameters.keras_profile'),
            backend=param.get_path('learner.parameters.backend'),
            device=param.get_path('learner.parameters.device'),
            verbose=False
        )

        # encoder used to convert text labels into vector
        many_hot_encoder = dcase_util.data.ManyHotEncoder(
                label_list=db.tags(),
                time_resolution=1
            )

        # =====================================================================
        # Training first pass
        # =====================================================================

        fold = 1
        # Get model filename
        fold1_model_filename = os.path.join(
            param.get_path('path.application.learner'),
            'model_fold_{fold}.h5'.format(fold=fold)
        )

        if not os.path.isfile(fold1_model_filename) or overwrite_learning:
            # Split the dataset into training and validation files
            training_files, validation_files = db.validation_split(
                fold=fold,
                split_type='random',
                validation_amount=param.get_path('learner.parameters.model.first_pass.validation_amount'),
                verbose=True
            )

            batch_size = param.get_path('learner.parameters.model.first_pass.fit.batch_size')
            shuffle = param.get_path('learner.parameters.model.first_pass.fit.shuffle')

            # Get items (with labels) associated with training files
            training_items = db.train(fold=fold).filter(file_list=training_files)

            # Create the generator, which convert filename and item into arrays batch_X, batch_y in right formats
            training_generator = data_generator(training_items, param.get_path('path.application.feature_extractor'),
                                                many_hot_encoder, feature_processing_chain,
                                                batch_size=batch_size, shuffle=shuffle)

            validation_items = db.train(fold=fold).filter(file_list=validation_files)
            validation_generator = data_generator(validation_items, param.get_path('path.application.feature_extractor'),
                                                  many_hot_encoder, feature_processing_chain,
                                                  batch_size=batch_size, shuffle=False)

            # Update constants with useful information to setup the model
            model_parameter_constants = {
                'NB_CLASSES': db.tag_count(),
                'INPUT_FREQUENCIES': param.get_path('feature_extractor.parameters.mel.n_mels'),
                'INPUT_SEQUENCE_LENGTH': param.get_path('feature_sequencer.sequence_length'),
            }
            model_parameter_constants.update(param.get_path('learner.parameters.model.constants', {}))

            # Load the sequential keras model defined in the YAML.
            keras_model_first_pass = dcase_util.keras.create_sequential_model(
                model_parameter_list=param.get_path('learner.parameters.model.first_pass.config'),
                constants=model_parameter_constants
            )

            # Print the model configuration
            keras_model_first_pass.summary(print_fn=log.line)

            # Create optimizer object from info given in YAML
            param.set_path(
                path='learner.parameters.compile.optimizer',
                new_value=dcase_util.keras.create_optimizer(
                    class_name=param.get_path('learner.parameters.optimizer.class_name'),
                    config=param.get_path('learner.parameters.optimizer.config')
                )
            )
            # Compile model
            keras_model_first_pass.compile(
                **param.get_path('learner.parameters.compile')
            )

            epochs = param.get_path('learner.parameters.model.first_pass.fit.epochs')

            # Setup callbacks used during training
            callback_list = [
                dcase_util.keras.ProgressLoggerCallback(
                    epochs=epochs,
                    metric=param.get_path('learner.parameters.compile.metrics')[0],
                    loss=param.get_path('learner.parameters.compile.loss'),
                    output_type='logging',
                    **param.get_path('learner.parameters.callbacks.ProgressLoggerCallback')
                )
            ]
            if param.get_path('learner.parameters.callbacks.StopperCallback'):
                callback_list.append(
                    dcase_util.keras.StopperCallback(
                        epochs=epochs,
                        **param.get_path('learner.parameters.callbacks.StopperCallback')
                    )
                )

            if param.get_path('learner.parameters.callbacks.StasherCallback'):
                callback_list.append(
                    dcase_util.keras.StasherCallback(
                        epochs=epochs,
                        **param.get_path('learner.parameters.callbacks.StasherCallback')
                    )
                )

            processing_interval = param.get_path(
                'learner.parameters.callbacks.ProgressLoggerCallback.processing_interval'
            )
            epochs = param.get_path('learner.parameters.model.first_pass.fit.epochs')

            # Iterate through epoch to be able to manually update callbacks
            for epoch_start in range(0, epochs, processing_interval):
                epoch_end = epoch_start + processing_interval

                # Make sure we have only specified amount of epochs
                if epoch_end > epochs:
                    epoch_end = epochs

                # Train keras_model_first_pass
                keras_model_first_pass.fit_generator(
                    generator=training_generator,
                    steps_per_epoch=len(training_files) // batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_files) // batch_size,
                    callbacks=callback_list,
                    verbose=0,
                    initial_epoch=epoch_start,
                    epochs=epoch_end
                )

                # Get f_measures of the current epoch
                val_macro_f_measure = get_f_measure_by_class(keras_model_first_pass, db.tag_count(), validation_generator,
                                                             len(validation_files) // batch_size)
                val_macro_f_measure = val_macro_f_measure.mean()

                tra_macro_f_measure = get_f_measure_by_class(keras_model_first_pass, db.tag_count(), training_generator,
                                                             len(training_files) // batch_size,
                                                             )
                tra_macro_f_measure = tra_macro_f_measure.mean()

                # Inject external metric values to the callbacks
                for callback in callback_list:
                    if hasattr(callback, 'set_external_metric_value'):
                        callback.set_external_metric_value(
                            metric_label='val_macro_f_measure',
                            metric_value=val_macro_f_measure
                        )
                        callback.set_external_metric_value(
                            metric_label='tra_macro_f_measure',
                            metric_value=tra_macro_f_measure
                        )

                # Manually update callbacks
                for callback in callback_list:
                    if hasattr(callback, 'update'):
                        callback.update()

                # Check we need to stop training
                stop_training = False
                for callback in callback_list:
                    if hasattr(callback, 'stop'):
                        if callback.stop():
                            log.line("Early stropping")
                            stop_training = True

                if stop_training:
                    # Stop the training loop
                    break

            # Fetch best model
            for callback in callback_list:
                if isinstance(callback, dcase_util.keras.StasherCallback):
                    callback.log()
                    best_weights = callback.get_best()['weights']
                    if best_weights:
                        keras_model_first_pass.set_weights(best_weights)
                    break

            # Save trained model
            keras_model_first_pass.save(fold1_model_filename)

            log.foot()

        # =======
        # Calculate best thresholds
        # =======
        thresholds_filename = os.path.join(
            param.get_path('path.application.learner'),
            'thresholds_{fold}.p'.format(fold=fold)
        )

        if not os.path.isfile(thresholds_filename) or overwrite_learning:
            training_files, validation_files = db.validation_split(
                fold=fold,
                split_type='random',
                validation_amount=param.get_path('learner.parameters.model.first_pass.validation_amount'),
                verbose=True
            )
            batch_size = param.get_path('learner.parameters.model.first_pass.fit.batch_size')
            validation_items = db.train(fold=fold).filter(file_list=validation_files)
            validation_generator = data_generator(validation_items, param.get_path('path.application.feature_extractor'),
                                                  many_hot_encoder, feature_processing_chain,
                                                  batch_size=batch_size, shuffle=False)

            # Load model if not trained during this run
            if not keras_model_first_pass:
                keras_model_first_pass = keras.models.load_model(fold1_model_filename)

            thresholds = [0] * db.tag_count()
            max_f_measure = [-numpy.inf] * db.tag_count()
            for threshold in numpy.arange(0., 1 + 1e-6, 0.1):
                # Assign current threshold to each class
                current_thresholds = [threshold] * db.tag_count()

                # Calculate f_measures with the current thresholds
                macro_f_measure = get_f_measure_by_class(keras_model_first_pass, db.tag_count(), validation_generator,
                                                         len(validation_files) // batch_size,
                                                         current_thresholds)

                # Update thresholds for class with better f_measures
                for i, label in enumerate(db.tags()):
                    f_measure = macro_f_measure[i]
                    if f_measure > max_f_measure[i]:
                        max_f_measure[i] = f_measure
                        thresholds[i] = threshold

            for i, label in enumerate(db.tags()):
                log.line("{:30}, threshold: {}".format(label, thresholds[i]))

            thresholds_filename = os.path.join(
                param.get_path('path.application.learner'),
                'thresholds.p'.format(fold=fold)
            )
            pickle.dump(thresholds, open(thresholds_filename, "wb"))

        else:
            thresholds = pickle.load(open(thresholds_filename, "rb"))

        # =====================================================================
        # Predict stage from weak to predict unlabel_in_domain tags
        # =====================================================================

        log.section_header('Predict 1st pass, add labels to unlabel_in_domain data')

        # Get results filename
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'pred_weak_fold_{fold}.txt'.format(fold=fold)
        )

        if not os.path.isfile(fold_results_filename) or overwrite_testing:
            # Initialize results container
            res = dcase_util.containers.MetaDataContainer(
                filename=fold_results_filename
            )

            # Load model if not yet loaded
            if not keras_model_first_pass:
                keras_model_first_pass = keras.models.load_model(fold1_model_filename)

            # Loop through all test files from the current cross-validation fold
            for item in db.test(fold=fold):
                # Get feature filename
                feature_filename = dcase_util.utils.Path(
                    path=item.filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                features = feature_processing_chain.process(
                    filename=feature_filename
                )

                input_data = features.data.reshape(features.shape[:-1]).T  # (500, 64)
                input_data = input_data.reshape((1,)+input_data.shape)  # (1, 500, 64)

                # Get network output
                probabilities = keras_model_first_pass.predict(x=input_data)

                # Binarization of the network output
                frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                    probabilities=probabilities,
                    binarization_type='class_threshold',
                    threshold=thresholds,
                    time_axis=0
                )

                estimated_tags = dcase_util.data.DecisionEncoder(
                    label_list=db.tags()
                ).many_hot(
                    frame_decisions=frame_decisions,
                    time_axis=0
                )

                # Store result into results container
                res.append(
                    {
                        'filename': item.filename,
                        'tags': estimated_tags[0]
                    }
                )

            # Save results container
            res.save()

        log.foot()

        # =====================================================================
        # Learning stage 2nd pass, learn from weak and unlabel_in_domain annotated data
        # =====================================================================

        fold = 2

        log.line(data='Fold [{fold}]'.format(fold=fold), indent=2)

        # Get model filename
        fold2_model_filename = os.path.join(
            param.get_path('path.application.learner'),
            'model_fold_{fold}.h5'.format(fold=fold)
        )

        if not os.path.isfile(fold2_model_filename) or overwrite_learning:

            model_parameter_constants = {
                'NB_CLASSES': db.tag_count(),
                'INPUT_FREQUENCIES': param.get_path('feature_extractor.parameters.mel.n_mels'),
                'INPUT_SEQUENCE_LENGTH': param.get_path('feature_sequencer.sequence_length'),
            }
            model_parameter_constants.update(param.get_path('learner.parameters.model.constants', {}))

            keras_model_second_pass = dcase_util.keras.create_sequential_model(
                model_parameter_list=param.get_path('learner.parameters.model.second_pass.config'),
                constants=model_parameter_constants
            )

            keras_model_second_pass.summary(print_fn=log.line)

            # Create optimizer object
            param.set_path(
                path='learner.parameters.compile.optimizer',
                new_value=dcase_util.keras.create_optimizer(
                    class_name=param.get_path('learner.parameters.optimizer.class_name'),
                    config=param.get_path('learner.parameters.optimizer.config')
                )
            )
            # Compile model
            keras_model_second_pass.compile(
                **param.get_path('learner.parameters.compile')
            )

            # Get annotations from the 1st pass model
            fold1_results_filename = os.path.join(
                param.get_path('path.application.recognizer'),
                'pred_weak_fold_{fold}.txt'.format(fold=1)
            )
            # Load annotations
            predictions_first_pass = dcase_util.containers.MetaDataContainer(
                filename=fold1_results_filename
            ).load()

            # Split the dataset into train and validation. If "weak" is provided, files from weak.csv are used to
            # validate the model. Else, give a percentage which will be used
            if param.get_path('learner.parameters.model.second_pass.validation_amount') == "weak":
                training_files = predictions_first_pass.unique_files
                training_items = predictions_first_pass
                validation_files = db.train(fold=1).unique_files
                validation_items = db.train(fold=1)
            else:
                # Get validation files
                training_files, validation_files = db.validation_split(
                    fold=fold,
                    split_type='random',
                    validation_amount=param.get_path('learner.parameters.model.second_pass.validation_amount'),
                    verbose=False
                )
                training_fold2 = predictions_first_pass + db.train(fold=1)

                training_items = training_fold2.filter(file_list=training_files)
                validation_items = training_fold2.filter(file_list=validation_files)

            processing_interval = param.get_path(
                'learner.parameters.callbacks.ProgressLoggerCallback.processing_interval'
            )
            epochs = param.get_path('learner.parameters.model.second_pass.fit.epochs')

            batch_size = param.get_path('learner.parameters.model.second_pass.fit.batch_size')
            shuffle = param.get_path('learner.parameters.model.second_pass.fit.shuffle')

            # Create generators, which convert filename and item into arrays batch_X, batch_y in right formats
            training_generator = data_generator(training_items, param.get_path('path.application.feature_extractor'),
                                                many_hot_encoder, feature_processing_chain,
                                                batch_size=batch_size, shuffle=shuffle, mode="strong")

            validation_generator = data_generator(validation_items, param.get_path('path.application.feature_extractor'),
                                                  many_hot_encoder,
                                                  feature_processing_chain,
                                                  batch_size=batch_size, shuffle=False, mode="strong")

            # Initialize callbacks used during training
            callback_list = [
                dcase_util.keras.ProgressLoggerCallback(
                    epochs=param.get_path('learner.parameters.model.second_pass.fit.epochs'),
                    metric=param.get_path('learner.parameters.compile.metrics')[0],
                    loss=param.get_path('learner.parameters.compile.loss'),
                    output_type='logging',
                    **param.get_path('learner.parameters.callbacks.ProgressLoggerCallback')
                )
            ]
            if param.get_path('learner.parameters.callbacks.StopperCallback'):
                callback_list.append(
                    dcase_util.keras.StopperCallback(
                        epochs=param.get_path('learner.parameters.model.second_pass.fit.epochs'),
                        **param.get_path('learner.parameters.callbacks.StopperCallback')
                    )
                )

            if param.get_path('learner.parameters.callbacks.StasherCallback'):
                callback_list.append(
                    dcase_util.keras.StasherCallback(
                        epochs=param.get_path('learner.parameters.model.second_pass.fit.epochs'),
                        **param.get_path('learner.parameters.callbacks.StasherCallback')
                    )
                )

            for epoch_start in range(0, epochs, processing_interval):
                epoch_end = epoch_start + processing_interval

                # Make sure we have only specified amount of epochs
                if epoch_end > epochs:
                    epoch_end = epochs

                # Train keras_model_second_pass
                keras_model_second_pass.fit_generator(
                    generator=training_generator,
                    steps_per_epoch=len(training_files) // batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_files) // batch_size,
                    callbacks=callback_list,
                    verbose=0,
                    initial_epoch=epoch_start,
                    epochs=epoch_end
                )

                # Calculate external metrics, f_measure of the current epoch
                val_macro_f_measure = get_f_measure_by_class(keras_model_second_pass, db.tag_count(), validation_generator,
                                                             len(validation_files) // batch_size, )
                val_macro_f_measure = val_macro_f_measure.mean()

                tra_macro_f_measure = get_f_measure_by_class(keras_model_second_pass, db.tag_count(), training_generator,
                                                             len(training_files) // batch_size,
                                                             )
                tra_macro_f_measure = tra_macro_f_measure.mean()

                # Inject external metric values to the callbacks
                for callback in callback_list:
                    if hasattr(callback, 'set_external_metric_value'):
                        callback.set_external_metric_value(
                            metric_label='val_macro_f_measure',
                            metric_value=val_macro_f_measure
                        )
                        callback.set_external_metric_value(
                            metric_label='tra_macro_f_measure',
                            metric_value=tra_macro_f_measure
                        )

                # Manually update callbacks
                for callback in callback_list:
                    if hasattr(callback, 'update'):
                        callback.update()

                # Check we need to stop training
                stop_training = False
                for callback in callback_list:
                    if hasattr(callback, 'stop'):
                        if callback.stop():
                            log.line("Early stropping")
                            stop_training = True

                if stop_training:
                    # Stop the training loop
                    break

            # Fetch best model
            for callback in callback_list:
                if isinstance(callback, dcase_util.keras.StasherCallback):
                    callback.log()
                    best_weights = callback.get_best()['weights']
                    if best_weights:
                        keras_model_second_pass.set_weights(best_weights)
                    break

            # Save trained model
            keras_model_second_pass.save(fold2_model_filename)

            log.foot()

    # =====================================================================
    # Testing stage, get strong annotations
    # =====================================================================

    if param.get_path('flow.testing'):
        log.section_header('Testing')

        # Get results filename
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.txt'.format(fold=2)
        )

        # Get model filename
        fold2_model_filename = os.path.join(
            param.get_path('path.application.learner'),
            'model_fold_{fold}.h5'.format(fold=2)
        )

        if not os.path.isfile(fold_results_filename) or overwrite_testing:
            # Load model if not yet loaded
            if not keras_model_second_pass:
                keras_model_second_pass = keras.models.load_model(fold2_model_filename)

            # Initialize results container
            res = dcase_util.containers.MetaDataContainer(
                filename=fold_results_filename
            )

            # Loop through all test files from the current cross-validation fold
            for item in db.test(fold=2):
                # Get feature filename
                feature_filename = dcase_util.utils.Path(
                    path=item.filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                # Get features array
                features = feature_processing_chain.process(
                    filename=feature_filename
                )

                input_data = features.data.reshape(features.shape[:-1]).T  # (500, 64)
                # Create a batch with only one file
                input_data = input_data.reshape((1,) + input_data.shape) # (1, 500, 64)

                # Get network output for strong data
                probabilities = keras_model_second_pass.predict(input_data)

                # only one file in the batch
                probabilities = probabilities[0]

                if param.get_path('recognizer.frame_binarization.enable'):
                    # Binarization of the network output
                    frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                        probabilities=probabilities,
                        binarization_type=param.get_path('recognizer.frame_binarization.binarization_type'),
                        threshold=param.get_path('recognizer.frame_binarization.threshold'),
                        time_axis=0
                    )
                else:
                    frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                        probabilities=probabilities,
                        binarization_type="global_threshold",
                        threshold=0.5,
                        time_axis=0
                    )

                decision_encoder = dcase_util.data.DecisionEncoder(
                    label_list=db.tags()
                )

                if param.get_path('recognizer.process_activity.enable'):
                    frame_decisions = decision_encoder.process_activity(
                        frame_decisions,
                        window_length=param.get_path('recognizer.process_activity.window_length'),
                        time_axis=0)

                for i, label in enumerate(db.tags()):

                    # given a list of ones, give the onset and offset in frames
                    estimated_events = decision_encoder.find_contiguous_regions(
                        activity_array=frame_decisions[:, i]
                    )

                    for [onset, offset] in estimated_events:
                        hop_length_seconds = param.get_path('feature_extractor.hop_length_seconds')
                        # Store result into results container, convert frames to seconds
                        res.append(
                            {
                                'filename': item.filename,
                                'event_label': label,
                                'onset': onset * hop_length_seconds,
                                'offset': offset * hop_length_seconds
                            }
                        )

            # Save results container
            res.save()
        log.foot()

    # =====================================================================
    # Evaluation stage, get results
    # =====================================================================

    if param.get_path('flow.evaluation'):
        log.section_header('Evaluation')

        stats_filename = os.path.join(param.get_path('path.application.recognizer'), 'evaluation.txt')

        if not os.path.isfile(stats_filename) or overwrite_testing:
            fold_results_filename = os.path.join(
                param.get_path('path.application.recognizer'),
                'res_fold_{fold}.txt'.format(fold=fold)
            )

            # test data used to evaluate the system
            reference_event_list = db.eval(fold=fold)

            # predictions done during the step test before
            estimated_event_list = dcase_util.containers.MetaDataContainer().load(
                filename=fold_results_filename
            )

            # Calculate the metric
            event_based_metric = event_based_evaluation(reference_event_list, estimated_event_list)

            with open(stats_filename, "w") as stats_file:
                stats_file.write(event_based_metric.__str__())

            log.line(event_based_metric.__str__(), indent=4)

        log.foot()


def data_generator(items, feature_path, many_hot_encoder, feature_processing_chain, batch_size=1, shuffle=True, mode='weak'):
    """ Transform MetaDataContainer into batches of data

    Parameters
    ----------

    items : MetaDataContainer, items to be generated

    feature_path : String, base path where features are stored

    many_hot_encoder : ManyHotEncoder, class to encode data

    feature_processing_chain : ProcessingChain, chain to process data

    batch_size : int, size of the batch to be returned

    shuffle : bool, shuffle the items before creating the batch

    mode : "weak" or "strong", indicate to return labels as tags (1/file) or event_labels (1/frame)

    Return
    ------

    (batch_X, batch_y): generator, arrays containing batches of data.

    """
    while True:
        batch_X = []
        batch_y = []
        if shuffle:
            random.shuffle(items)
        for item in items:
            # Get feature filename
            feature_filename = dcase_util.utils.Path(
                path=item.filename
            ).modify(
                path_base=feature_path,
                filename_extension='.cpickle',
            )

            features = feature_processing_chain.process(
                filename=feature_filename
            )
            input_data = features.data.reshape(features.shape[:-1]).T

            # Target
            targets = item.tags
            targets = many_hot_encoder.encode(targets, length_frames=1).data.flatten()
            if mode == "strong":
                targets = numpy.repeat(targets.reshape((1,) + targets.shape), input_data.shape[0], axis=0)

            if batch_size == 1:
                batch_X = input_data.reshape((1,) + input_data.shape)
                batch_y = targets.reshape((1,) + targets.shape)
            else:
                batch_X.append(input_data)
                batch_y.append(targets)
                if len(batch_X) == batch_size and len(batch_y) == batch_size:
                    yield numpy.array(batch_X), numpy.array(batch_y)

                    batch_X = []
                    batch_y = []



if __name__ == "__main__":
    # Read parameters file
    parameters = dcase_util.containers.DictContainer().load(
        filename='task4_crnn.yaml'
    )

    try:
        sys.exit(main(parameters))
    except (ValueError, IOError) as e:
        sys.exit(e)
