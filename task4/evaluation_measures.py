# !/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# Initial software, Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah
# Copyright © INRIA, 2018, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

from dcase_util.data import ProbabilityEncoder
import sed_eval
import numpy
import pandas as pd

def get_f_measure_by_class(keras_model, nb_tags, generator, steps, thresholds=None):
    """ get f measure for each class given a model and a generator of data (X, y)

    Parameters
    ----------

    keras_model : Model, model to get predictions

    nb_tags : int, number of classes which are represented

    generator : generator, data generator used to get f_measure

    steps : int, number of steps the generator will be used before stopping

    thresholds : int or list, thresholds to apply to each class to binarize probabilities

    Return
    ------

    macro_f_measure : list, f measure for each class

    """

    # Calculate external metrics
    TP = numpy.zeros(nb_tags)
    TN = numpy.zeros(nb_tags)
    FP = numpy.zeros(nb_tags)
    FN = numpy.zeros(nb_tags)
    for counter, (X, y) in enumerate(generator):
        if counter == steps:
            break
        predictions = keras_model.predict(X)

        if len(predictions.shape) == 3:
            # average data to have weak labels
            predictions = numpy.mean(predictions, axis=1)
            y = numpy.mean(y, axis=1)

        if thresholds is None:
            binarization_type = 'global_threshold'
            thresh = 0.5
        else:
            binarization_type = "class_threshold"
            assert type(thresholds) is list
            thresh = thresholds

        predictions = ProbabilityEncoder().binarization(predictions,
                                                        binarization_type=binarization_type,
                                                        threshold=thresh,
                                                        time_axis=0
                                                        )

        TP += (predictions + y == 2).sum(axis=0)
        FP += (predictions - y == 1).sum(axis=0)
        FN += (y - predictions == 1).sum(axis=0)
        TN += (predictions + y == 0).sum(axis=0)

    macro_f_measure = numpy.zeros(nb_tags)
    mask_f_score = 2*TP + FP + FN != 0
    macro_f_measure[mask_f_score] = 2*TP[mask_f_score] / (2*TP + FP + FN)[mask_f_score]

    return macro_f_measure


def event_based_evaluation(reference_event_list, estimated_event_list):
    """ Calculate sed_eval event based metric for challenge

        Parameters
        ----------

        reference_event_list : MetaDataContainer, list of referenced events

        estimated_event_list : MetaDataContainer, list of estimated events

        Return
        ------

        event_based_metric : EventBasedMetrics

        """

    files = {}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=0.200,
        percentage_of_length=0.2,
        empty_system_output_handling='zero_score'
    )

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        # events = []
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                # events.append(event.event_label)
        estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return event_based_metric


def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    :param df: pd.DataFrame, the dataframe to search on
    :param fname: the filename to extract the value from the dataframe
    :return: list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"].str.contains(fname)]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict('records')
    else:
        event_list_for_current_file = event_file.to_dict('records')

    return event_list_for_current_file

def event_based_evaluation_df(reference, estimated):
    """
    Calculate EventBasedMetric given a reference and estimated dataframe
    :param reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
    reference events
    :param estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
    estimated events to be compared with reference
    :return: sed_eval.sound_event.EventBasedMetrics with the scores
    """
    
    evaluated_files = reference["filename"].unique()
    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=0.200,
        percentage_of_length=0.2,
        empty_system_output_handling='zero_score'
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(estimated, fname)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )
    return event_based_metric
