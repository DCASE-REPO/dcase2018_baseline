#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dcase_util
import sys
import os
import argparse
import textwrap


def handle_application_arguments(param, application_title='', version=''):
    """Handle application arguments

    Parameters
    ----------
    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    application_title : str
        Application title
        Default value ''

    version : str
        Application version
        Default value ''

    Returns
    -------
    nothing


    """

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            '''\
            DCASE 2018 
            {app_title}
            Baseline system
            ---------------------------------------------            
            Author:  Toni Heittola ( toni.heittola@tut.fi )
            Tampere University of Technology / Audio Research Group
            '''.format(app_title=application_title)
        )
    )

    # Setup argument handling
    parser.add_argument(
        '-m', '--mode',
        choices=('dev', 'eval', 'leaderboard'),
        default=None,
        help="Selector for application operation mode",
        required=False,
        dest='mode',
        type=str
    )

    # Application parameter modification
    parser.add_argument(
        '-s', '--parameter_set',
        help='Parameter set id, can be comma separated list',
        dest='parameter_set',
        required=False,
        type=str
    )

    parser.add_argument(
        '-p', '--param_file',
        help='Parameter file override',
        dest='parameter_override',
        required=False,
        metavar='FILE',
        type=dcase_util.utils.argument_file_exists
    )

    # Specific actions
    parser.add_argument(
        '--overwrite',
        help='Overwrite mode',
        dest='overwrite',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--download_dataset',
        help='Download dataset to given path and exit',
        dest='dataset_path',
        required=False,
        type=str
    )

    # Output
    parser.add_argument(
        '-o', '--output',
        help='Output file',
        dest='output_file',
        required=False,
        type=str
    )

    # Show information
    parser.add_argument(
        '--show_parameters',
        help='Show active application parameter set',
        dest='show_parameters',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--show_sets',
        help='List of available parameter sets',
        dest='show_set_list',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--show_results',
        help='Show results of the evaluated system setups',
        dest='show_results',
        action='store_true',
        required=False
    )

    # Application information
    parser.add_argument(
        '-v', '--version',
        help='Show version number and exit',
        action='version',
        version='%(prog)s ' + version
    )

    # Parse arguments
    args = parser.parse_args()

    if args.parameter_override:
        # Override parameters from a file
        param.override(override=args.parameter_override)

    if args.overwrite:
        # Inject overwrite into parameters
        param['general']['overwrite'] = True

    if args.show_parameters:
        # Process parameters, and clean up parameters a bit for showing
        param_ = dcase_util.containers.AppParameterContainer(param).process(
            create_paths=False,
            create_parameter_hints=False
        )

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


def save_system_output(db, folds, param, log, output_file, mode='dcase'):
    """Save system output

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

    output_file : str

    mode : str
        Output mode, possible values ['dcase', 'leaderboard']
        Default value 'dcase'

    Returns
    -------
    nothing

    """

    # Initialize results container
    all_res = dcase_util.containers.MetaDataContainer(
        filename=output_file
    )

    # Loop over all cross-validation folds and collect results
    for fold in folds:
        # Get results filename
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.txt'.format(fold=fold)
        )

        if os.path.isfile(fold_results_filename):
            # Load results container
            res = dcase_util.containers.MetaDataContainer().load(
                filename=fold_results_filename
            )
            all_res += res

        else:
            raise ValueError(
                'Results output file does not exists [{fold_results_filename}]'.format(
                    fold_results_filename=fold_results_filename
                )
            )

    # Convert paths to relative to the dataset root
    for item in all_res:
        item.filename = db.absolute_to_relative_path(item.filename)

        if mode == 'leaderboard':
            item['Id'] = os.path.splitext(os.path.split(item.filename)[-1])[0]
            item['Scene_label'] = item.scene_label

    if mode == 'leaderboard':
        all_res.save(fields=['Id', 'Scene_label'], delimiter=',')

    else:
        all_res.save(csv_header=False)

    log.line('System output saved to [{output_file}]'.format(output_file=output_file), indent=2)
    log.line()


def show_general_information(parameter_set, active_folds, param, db, log):
    """Show application general information

    Parameters
    ----------
    parameter_set : str
        Dataset

    active_folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    db : dcase_util.dataset.Dataset
        Dataset

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    log.section_header('General information')
    log.line('Parameter set', indent=2)
    log.data(field='Set ID', value=parameter_set, indent=4)
    log.data(field='Set description', value=param.get('description'), indent=4)

    log.line('Application', indent=2)
    log.data(field='Overwrite', value=param.get_path('general.overwrite'), indent=4)

    log.data(field='Dataset', value=db.storage_name, indent=4)
    log.data(field='Active folds', value=active_folds, indent=4)
    log.line()
    log.foot()


def show_results(param, log):
    """Show system evaluation results

    Parameters
    ----------
    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    eval_path = param.get_path('path.application.evaluator')

    eval_files = dcase_util.utils.Path().file_list(path=eval_path, extensions='yaml')

    eval_data = {}
    for filename in eval_files:
        data = dcase_util.containers.DictContainer().load(filename=filename)
        set_id = data.get_path('parameters.set_id')
        if set_id not in eval_data:
            eval_data[set_id] = {}

        params_hash = data.get_path('parameters._hash')

        if params_hash not in eval_data[set_id]:
            eval_data[set_id][params_hash] = data

    log.section_header('Evaluated systems')
    log.line()
    log.row_reset()
    log.row(
        'Set ID', 'Mode', 'Accuracy', 'Description', 'Parameter hash',
        widths=[25, 10, 11, 45, 35],
        separators=[False, True, True, True, True],
        types=['str25', 'str10', 'float1_percentage', 'str', 'str']
    )
    log.row_sep()
    for set_id in sorted(list(eval_data.keys())):
        for params_hash in eval_data[set_id]:
            data = eval_data[set_id][params_hash]
            desc = data.get_path('parameters.description')
            application_mode = data.get_path('application_mode', '')
            log.row(
                set_id,
                application_mode,
                data.get_path('overall_accuracy') * 100.0,
                desc,
                params_hash
            )
    log.line()
    sys.exit(0)


def show_parameter_sets(param, log):
    """Show available parameter sets

    Parameters
    ----------
    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    log.section_header('Parameter sets')
    log.line()
    log.row_reset()
    log.row(
        'Set ID', 'Description',
        widths=[20, 70],
        separators=[True, False],
    )
    log.row_sep()
    for set_id in param.set_ids():
        current_parameter_set = param.get_set(set_id=set_id)

        if current_parameter_set:
            desc = current_parameter_set.get('description', '')
        else:
            desc = ''

        log.row(
            set_id,
            desc
        )

    log.line()
