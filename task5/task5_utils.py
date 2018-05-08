import dcase_util
import numpy

# =====================================================================
# Function: get labels and feature filenames from train/val wav files
# =====================================================================
def get_label_from_filename(db,filenames_train,filenames_val,param):
    """get labels and feature filenames from train/val wav files

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    filenames_train : list of strings
        filenames of training set

    filenames_val : list of strings
        filenames of validation set

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    Returns
    -------
    filename_train_list : list
        feature filenames of training set
    label_train_list : list
        labels of training set
    filename_val_list : list
        feature filenames of validation set
    label_val_list : list
        labels of validation set

    """

    filename_train_list = []
    filename_val_list = []
    label_train_list = []
    label_val_list = []
    for item in db:
        # Get feature filename
        feature_filename = dcase_util.utils.Path(
            path=item.filename
        ).modify(
            path_base=param.get_path('path.application.feature_extractor'),
            filename_extension='.cpickle'
        )

        if item.filename in filenames_train:
            filename_train_list.append(feature_filename)
            label_train_list.append(item.scene_label)
        if item.filename in filenames_val:
            filename_val_list.append(feature_filename)
            label_val_list.append(item.scene_label)
    return filename_train_list,label_train_list,filename_val_list,label_val_list

# =====================================================================
# Function: get processing chain based on params
# =====================================================================
def get_processing_chain(param,fold_stats_filename=None,chain_type = 'data_processing_chain'):
    """get labels and feature filenames from train/val wav files

    Parameters
    ----------
    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    fold_stats_filename : str
        filenames of normalizer

    Returns
    -------
    data_processing_chain : dcase_util.processors.ProcessingChain
        Data processing chain (e.g. Reading, Normalization, ...)

    """

    processing_chain = dcase_util.processors.ProcessingChain()
    for chain in param.get_path(chain_type + '.parameters.chain'):
        processor_name = chain.get('processor_name')
        init_parameters = chain.get('init_parameters', {})

        # Inject parameters
        if processor_name == 'NormalizationProcessor':
            init_parameters['filename'] = fold_stats_filename

        if processor_name == 'AudioReadingProcessor':
            init_parameters['fs'] = param.get_path('feature_extractor.fs')

        if processor_name == 'RepositoryFeatureExtractorProcessor':
            init_parameters = param.get_path('feature_extractor')

        if init_parameters.get('enable') is None or init_parameters.get('enable') is True:
            processing_chain.push_processor(
                processor_name=processor_name,
                init_parameters=init_parameters,
            )
    return processing_chain