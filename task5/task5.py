#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DCASE 2018
# Task 5: Monitoring of domestic activities based on multi-channel acoustics
# Baseline system
# ---------------------------------------------
# Author:  Gert Dekkers ( gert.dekkers@kuleuven.be ) and Toni Heittola ( toni.heittola@tut.fi )
# KU Leuven University / Advanced Integrated Sensing lab (ADVISE)
# Tampere University of Technology / Audio Research Group

import dcase_util
import sys
import numpy
import os
import copy
import argparse
import textwrap
import pickle

from task5_utils import get_processing_chain, get_label_from_filename, save_system_output
import sklearn.utils as skutils
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

__version_info__ = ('3', '0', '0')
__version__ = '.'.join(__version_info__)

# =====================================================================
# Function: Handle cmdline args
# =====================================================================
def handle_application_arguments(param):
    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2018 
            Task 5: Monitoring of domestic activities based on multi-channel acoustics
            Baseline system
            ---------------------------------------------            
            Author:  Gert Dekkers ( gert.dekkers@kuleuven.be ) and Toni Heittola ( toni.heittola@tut.fi )
            KU Leuven University / Advanced Integrated Sensing lab (ADVISE)
            Tampere University of Technology / Audio Research Group
        '''))

    # Setup argument handling
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s ' + __version__
    )

    parser.add_argument(
        '-m', '--mode',
        choices=('dev', 'eval'),
        default=None,
        help="selector for application operation mode",
        required=False,
        dest='mode',
        type=str
    )

    parser.add_argument(
        '-o', '--overwrite',
        help='overwrite mode',
        dest='overwrite',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '-p', '--show_parameters',
        help='show active application parameter set',
        dest='show_parameters',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '-d', '--dataset',
        help='download dataset to given path and exit',
        dest='dataset_path',
        required=False,
        type=str
    )

    parser.add_argument(
        '-s', '--parameter_set',
        help='Parameter set id',
        dest='parameter_set',
        required=False,
        type=str
    )

    # Parse arguments
    args = parser.parse_args()

    if args.parameter_set:
        # Set parameter set
        param['active_set'] = args.parameter_set
        param.update_parameter_set(args.parameter_set)

    if args.overwrite:
        # Inject overwrite into parameters
        param['general']['overwrite'] = True

    if args.show_parameters:
        # Process parameters, and clean up parameters a bit for showing
        param_ = copy.deepcopy(param)

        del param_['sets']
        del param_['defaults']
        for section in param_:
            if section.endswith('_method_parameters'):
                param_[section] = {}

        param_.log()
        sys.exit(0)

    if args.dataset_path:
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

    return args

# =====================================================================
# Function: Feature extraction
# =====================================================================
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
        Default value False

    Returns
    -------
    nothing

    """

    # Define processing chain (Multichannel audio reading + feature extraction for each channel)
    feature_processing_chain = get_processing_chain(param, chain_type='feature_processing_chain')
    # Loop over all audio files in the current dataset and extract acoustic features for each of them.
    for audio_filename in db.audio_files:
        # Get filename for feature data from audio filename
        feature_filename = dcase_util.utils.Path(
            path=audio_filename
        ).modify(
            path_base=param.get_path('path.application.feature_extractor'),
            filename_extension='.cpickle'
        )

        if not os.path.isfile(feature_filename) or overwrite:
            log.line(
                data=os.path.split(audio_filename)[1],
                indent=2
            )
            # Extract features and store them into FeatureContainer, and save it to the disk
            feature_processing_chain.process(filename=audio_filename).save(filename=feature_filename)

# =====================================================================
# Function: Get feature normalization params (µ,std)
# =====================================================================
def do_feature_normalization(db, folds, param, log, overwrite=False):
    """Feature normalization stage

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

    # Loop over all active cross-validation folds and calculate mean and std for the training data
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
            normalizer = dcase_util.data.RepositoryNormalizer(
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

                # Load feature streams
                features = dcase_util.containers.FeatureRepository().load(filename=feature_filename)

                # Accumulate statistics (One mean/std for all channels)
                normalizer.accumulate(
                    data=features
                )

            # Finalize and save
            normalizer.finalize().save()

# =====================================================================
# Function: Learning
# =====================================================================
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
    for fold in folds:
        log.line(data='Fold [{fold}]'.format(fold=fold), indent=2)

        # Get model filename
        fold_model_filename = os.path.join(
            param.get_path('path.application.learner'),
            'model_fold_{fold}.h5'.format(fold=fold)
        )

        if not os.path.isfile(fold_model_filename) or overwrite:
            # -- prepare learner -- #
            # Setup keras, run only once
            dcase_util.keras.setup_keras(
                seed=param.get_path('learner.parameters.random_seed'),
                profile=param.get_path('learner.parameters.keras_profile'),
                backend=param.get_path('learner.parameters.backend')
            )
            import keras

            # Create model
            keras_model = dcase_util.keras.create_sequential_model(
                model_parameter_list=param.get_path('learner.parameters.model.config'),
                constants=param.get_path('learner.parameters.model.constants')
            )

            # Show model topology
            log.line(
                dcase_util.keras.model_summary_string(keras_model)
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
            keras_model.compile(**param.get_path('learner.parameters.compile'))

            # -- data related -- #
            # Get validation files
            validationsplit_fold_filename = os.path.join(param.get_path('path.application.learner'),'validationsplit_fold_{fold}.pickle'.format(fold=fold))
            if not os.path.isfile(validationsplit_fold_filename):
                training_files, validation_files = db.validation_split(
                    fold=fold,
                    split_type='balanced',
                    validation_amount=param.get_path('learner.parameters.validation_amount'),
                    verbose=True
                )
                with open(validationsplit_fold_filename, 'wb') as f: pickle.dump([training_files,validation_files], f)
            else:
                with open(validationsplit_fold_filename, "rb") as f: [training_files,validation_files] = pickle.load(f)  # load

            # get matching labels
            training_files, training_labels, validation_files, val_labels = get_label_from_filename(db.train(fold=fold), training_files, validation_files, param)

            # Get label encoder
            label2num_enc = LabelEncoder()
            scene_labels_num = label2num_enc.fit_transform(db.scene_labels())

            # convert labels to numeric format
            training_labels_num = label2num_enc.transform(training_labels)
            val_labels_num = label2num_enc.transform(val_labels)

            # get amount of batches for training and validation
            if param.get_path('learner.parameters.fit.undersampling'):
                class_weight = skutils.compute_class_weight('balanced', numpy.unique(training_labels_num),training_labels_num)  # get class weights
                train_batches = (sum(training_labels_num == numpy.argmax(class_weight)) * len(numpy.unique(training_labels_num))) // param.get_path('learner.parameters.fit.batch_size')
            else:
                train_batches = len(training_files) // param.get_path('learner.parameters.fit.batch_size')  # get amount of batches
            val_batches = int(numpy.ceil(len(validation_files)/param.get_path('learner.parameters.fit.batch_size')))  # get amount of batches

            # get normalizer filename
            fold_stats_filename = os.path.join(
                param.get_path('path.application.feature_normalizer'),
                'norm_fold_{fold}.cpickle'.format(fold=fold)
            )

            # Create data processing chain for features
            data_processing_chain = get_processing_chain(param, fold_stats_filename=fold_stats_filename, chain_type = 'data_processing_chain')

            # Init data generators
            from task5_datagenerator import DataGenerator
            TrainDataGenerator = DataGenerator(training_files, training_labels_num,
                                               data_processing_chain=data_processing_chain,
                                               batches=train_batches,
                                               batch_size=param.get_path('learner.parameters.fit.batch_size'),
                                               undersampling=param.get_path('learner.parameters.fit.undersampling'),
                                               shuffle=True)
            ValidationDataGenerator = DataGenerator(validation_files, val_labels_num,
                                               data_processing_chain=data_processing_chain,
                                               batches=val_batches,
                                               batch_size=param.get_path('learner.parameters.fit.batch_size'),
                                               undersampling='None',
                                               shuffle=False)

            # -- train/epoch loop -- #
            prevmodelload = False
            val_scores = []
            epoch_list = []
            for epoch_start in range(-1, param.get_path('learner.parameters.fit.epochs')-1,param.get_path('learner.parameters.fit.processing_interval')):  # for every epoch
                # update epoch information
                epoch_end = epoch_start + param.get_path('learner.parameters.fit.processing_interval')  # specifiy epoch range
                if epoch_end > param.get_path('learner.parameters.fit.epochs'):  # Make sure we have only specified amount of epochs
                    epoch_end = param.get_path('learner.parameters.fit.epochs') - 1

                model_fold_epoch_filename = os.path.join(param.get_path('path.application.learner'),'model_fold_{fold}_epoch_{epoch:d}.h5'.format(fold=fold,epoch=epoch_end))
                val_fold_epoch_filename = os.path.join(param.get_path('path.application.learner'),'val_fold_{fold}_epoch_{epoch:d}.pickle'.format(fold=fold,epoch=epoch_end))
                if not (os.path.isfile(model_fold_epoch_filename) & os.path.isfile(val_fold_epoch_filename)):  # if model does not exist
                    if prevmodelload:  # if epoch already performed before
                        log.line('Loaded model of fold {fold} epoch {epoch_start:d}'.format(fold=fold,epoch_start=epoch_start), indent=2)
                        keras_model = keras.models.load_model(prev_model_fold_epoch_filename)  # get model
                        prevmodelload = False

                    # train model
                    keras_model.fit_generator(
                        generator=TrainDataGenerator,
                        initial_epoch=epoch_start,
                        epochs=epoch_end,
                        steps_per_epoch=train_batches,
                        verbose=0
                    )

                    # evaluate model on validation set for each mic in each example, output is 4 channels * len(validation_files) long
                    posteriors_all = keras_model.predict_generator(
                        generator=ValidationDataGenerator,
                        steps=val_batches
                    )

                    # combine estimates/posteriors from different channels in a particular sensor node
                    posteriors = numpy.empty((len(validation_files),len(db.scene_labels())))
                    for i in range(len(validation_files)): # for each validation file
                        # mean rule
                        if param.get_path('recognizer.fusion.method')=='mean':
                            posteriors[i,:] = numpy.mean(posteriors_all[i*param.get_path('feature_extractor.channels'):(i+1)*param.get_path('feature_extractor.channels'),:],axis=0)
                        # none
                        elif param.get_path('recognizer.fusion.method')=='none':
                            posteriors = posteriors_all
                        else:
                            raise ValueError("Fusion not supported")

                    # get estimated labels
                    val_labels_est_num = numpy.argmax(posteriors, axis=1)

                    # get score
                    F1_score = f1_score(val_labels_num, val_labels_est_num, labels = scene_labels_num, average='macro')
                    log.line('Fold {fold} - Epoch {epoch:d}/{epochs:d} - validation set F1-score: {Fscore:f}'.format(fold=fold,epoch=epoch_end,epochs=param.get_path('learner.parameters.fit.epochs') - 1,Fscore=F1_score),indent=2)

                    # save intermediate results
                    keras_model.save(model_fold_epoch_filename)
                    with open(val_fold_epoch_filename, 'wb') as f: pickle.dump([F1_score], f)
                else: # if already performed
                    # update model loading and load performance prev model
                    prevmodelload = True
                    prev_model_fold_epoch_filename = model_fold_epoch_filename
                    with open(val_fold_epoch_filename, "rb") as f: [F1_score] = pickle.load(f)  # load

                # append scores
                val_scores.append(F1_score)
                epoch_list.append(epoch_end)

            # get best model
            epoch_best = epoch_list[numpy.argmax(val_scores)]
            log.line('Best performing model on epoch {epoch_best:d} with an F1-score of {Fscore:f}%'.format(epoch_best=epoch_best,Fscore=numpy.max(val_scores)*100), indent=2)
            # load best model
            model_fold_epoch_filename = os.path.join(param.get_path('path.application.learner'),'model_fold_{fold}_epoch_{epoch:d}.h5'.format(fold=fold,epoch=epoch_best))
            keras_model = keras.models.load_model(model_fold_epoch_filename)
            # save best model
            keras_model.save(fold_model_filename)

# =====================================================================
# Function: Testing
# =====================================================================
def do_testing(db, db_train, folds, param, log, overwrite=False):
    """Testing stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset (test)

    db_train : dcase_util.dataset.Dataset
        Dataset (train)

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

    # Loop over all cross-validation folds and test
    for fold in folds:
        log.line(
            data='Fold [{fold}]'.format(fold=fold),
            indent=2
        )

        # Setup keras, run only once
        dcase_util.keras.setup_keras(
            seed=param.get_path('learner.parameters.random_seed'),
            profile=param.get_path('learner.parameters.keras_profile'),
            backend=param.get_path('learner.parameters.backend')
        )
        import keras

        # Get results filename
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.txt'.format(fold=fold)
        )

        # check if results already processed
        if not os.path.isfile(fold_results_filename) or overwrite:
            # -- prepare learner -- #
            # Get model filename
            fold_model_filename = os.path.join(
                param.get_path('path.application.learner'),
                'model_fold_{fold}.h5'.format(fold=fold)
            )

            # load model
            keras_model = keras.models.load_model(fold_model_filename)

            # -- prepare data -- #
            # Get normalization factor filename
            fold_stats_filename = os.path.join(
                param.get_path('path.application.feature_normalizer'),
                'norm_fold_{fold}.cpickle'.format(fold=fold)
            )

            # Create processing chain for features
            data_processing_chain = get_processing_chain(param, fold_stats_filename=fold_stats_filename, chain_type='data_processing_chain')

            # Get label encoder
            label2num_enc = LabelEncoder()
            scene_labels_num = label2num_enc.fit_transform(db_train.scene_labels())

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

                # do processing chain
                features = data_processing_chain.process(filename=feature_filename)

                # Get network output
                posteriors_all = keras_model.predict(x=features.data)

                # combine estimates/posteriors from different channels in a particular sensor node
                if param.get_path('recognizer.fusion.method') == 'mean': # mean rule
                    posteriors = numpy.mean(posteriors_all, axis=0)
                elif param.get_path('recognizer.fusion.method') == 'none': # none (if channels are 'fed directly to/combined before the' classifier)
                    posteriors = posteriors_all
                else:
                    raise ValueError("Fusion not supported")

                estimated_scene_label = label2num_enc.inverse_transform(numpy.argmax(posteriors, axis=0))

                # Store result into results container
                res.append(
                    {
                        'filename': item.filename,
                        'scene_label': estimated_scene_label
                    }
                )

            # Save results container
            res.save()
# =====================================================================
# Function: Scoring
# =====================================================================
def do_evaluation(db, folds, param, log):
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

    Returns
    -------
    nothing

    """

    # get results for each fold
    class_wise_results = numpy.zeros((len(folds), len(db.scene_labels())))
    for i,fold in enumerate(folds):
        # filename results
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.txt'.format(fold=fold)
        )

        # get file labels
        reference_scene_label = []
        for item_id, item in enumerate(db.eval(fold=fold)):
            reference_scene_label.append(item.scene_label)
        estimated_scene_label = []
        for item_id, item in enumerate(dcase_util.containers.MetaDataContainer().load(filename=fold_results_filename)):
            estimated_scene_label.append(item.scene_label)

        # get class-wise scores
        class_wise_results[i,:] = f1_score(reference_scene_label, estimated_scene_label, labels=db.scene_labels(), average=None)

    # get overall score
    overall = numpy.mean(class_wise_results,axis=1)

    # Form results table
    cell_data = class_wise_results
    scene_mean_Fscore = numpy.mean(cell_data, axis=0).reshape((1, -1))
    cell_data = numpy.vstack((cell_data, scene_mean_Fscore))
    fold_mean_Fscore = numpy.mean(cell_data, axis=1).reshape((-1, 1))
    cell_data = numpy.hstack((cell_data, fold_mean_Fscore))
    scene_list = db.scene_labels()
    scene_list.extend(['Average'])
    cell_data = [scene_list] + (cell_data * 100.0).tolist()
    column_headers = ['Scene']
    for fold in folds:
        column_headers.append('Fold {fold}'.format(fold=fold))
    column_headers.append('Average')

    # show results
    log.table(
        cell_data=cell_data,
        column_headers=column_headers,
        column_separators=[0, 5],
        row_separators=[len(db.scene_labels())],
        indent=2
    )

# =====================================================================
# Main program
# =====================================================================
def main(argv):
    # Read parameters file
    parameters = dcase_util.containers.DictContainer().load(
        filename='task5.yaml'
    )

    # Process application parameters
    param = dcase_util.containers.DCASEAppParameterContainer(
        parameters,
        path_structure={
            'FEATURE_EXTRACTOR': ['FEATURE_PROCESSING_CHAIN'],
            'FEATURE_NORMALIZER': ['FEATURE_PROCESSING_CHAIN'],
            'LEARNER': ['FEATURE_PROCESSING_CHAIN','DATA_PROCESSING_CHAIN', 'LEARNER'],
            'RECOGNIZER': ['FEATURE_PROCESSING_CHAIN','DATA_PROCESSING_CHAIN', 'LEARNER', 'RECOGNIZER'],
        }
    )

    # Handle application arguments
    args = handle_application_arguments(param)

    # Process parameters
    param.process()

    # Make sure all system paths exists
    dcase_util.utils.Path().create(
        paths=list(param['path'].values())
    )

    # Setup logging
    dcase_util.utils.setup_logging(
        logging_file=os.path.join(param.get_path('path.log'), 'task5.log')
    )
    log = dcase_util.ui.ui.FancyLogger()
    log.title('DCASE2018 / Task5 -- Task 5: Monitoring of domestic activities based on multi-channel acoustics')
    log.line(' ')

    # Get dataset and initialize
    db = dcase_util.datasets.dataset_factory(
        dataset_class_name=param.get_path('dataset.parameters.dataset'),
        data_path=param.get_path('path.dataset'),
    ).initialize()

    # Get active folds
    if args.mode == 'eval' or param.get_path('general.eval_mode'):
        # Get active folds
        active_folds = db.folds(mode='full')
    else:
        # Get active folds
        active_folds = db.folds(mode=param.get_path('dataset.parameters.evaluation_mode'))
        active_fold_list = param.get_path('general.active_fold_list')
        if active_fold_list and len(set(active_folds).intersection(active_fold_list)) > 0:
            # Active fold list is set and it intersects with active_folds given by dataset class
            active_folds = list(set(active_folds).intersection(active_fold_list))

    if param.get_path('flow.feature_extraction'):
        # Feature extraction stage
        log.section_header('Feature Extraction')
        do_feature_extraction(
            db=db,
            param=param,
            log=log,
            overwrite=param.get_path('general.overwrite')
        )
        log.foot()

    if param.get_path('flow.feature_normalization'):
        # Feature normalization stage
        log.section_header('Feature Normalization')
        do_feature_normalization(
            db=db,
            folds=active_folds,
            param=param,
            log=log,
            overwrite=param.get_path('general.overwrite')
        )
        log.foot()

    if param.get_path('flow.learning'):
        # Learning stage
        log.section_header('Learning')
        do_learning(
            db=db,
            folds=active_folds,
            param=param,
            log=log,
            overwrite=False
        )
        log.foot()

    if ((not args.mode or args.mode == 'dev') & (not param.get_path('general.eval_mode'))):
        # System evaluation in "dev" mode
        if param.get_path('flow.testing'):
            # Testing stage
            log.section_header('Testing')
            do_testing(
                db=db,
                db_train=db,
                folds=active_folds,
                param=param,
                log=log,
                overwrite=param.get_path('general.overwrite')
            )
            log.foot()

        if param.get_path('flow.evaluation'):
            # Evaluation stage
            log.section_header('Evaluating')
            do_evaluation(
                db=db,
                folds=active_folds,
                param=param,
                log=log
            )
            log.foot()

    elif args.mode == 'eval' or param.get_path('general.eval_mode'):
        # updata param set
        param.update_parameter_set('eval')

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
            # Feature extraction stage for eval mode
            log.section_header('Feature Extraction')
            do_feature_extraction(
                db=db_eval,
                param=param,
                log=log,
                overwrite=param.get_path('general.overwrite')
            )
            log.foot()

        if param.get_path('flow.testing'):
            # Testing stage for eval mode
            log.section_header('Testing')
            do_testing(
                db=db_eval,
                db_train=db,
                folds=active_folds,
                param=param,
                log=log,
                overwrite=param.get_path('general.overwrite')
            )
            log.foot()

            save_system_output(
                db=db_eval,
                folds=active_folds,
                param=param,
                log=log,
                output_file='eval_output.txt',
                mode='dcase'
            )

        if db_eval.reference_data_present and param.get_path('flow.evaluation'):
            # Evaluation stage for eval mode
            log.section_header('Evaluating')
            do_evaluation(
                db=db_eval,
                folds=active_folds,
                param=param,
                log=log
            )
            log.foot()

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))

    except (ValueError, IOError) as e:
        sys.exit(e)