# DCASE 2018 - Task 5 - Baseline system

Authors:

- Gert Dekkers (<gert.dekkers@kuleuven.be>, <https://iiw.kuleuven.be/onderzoek/advise/People/Gert_Dekkers>)
- Peter Karsmakers (<peter.karsmakers@kuleuven.be>, <https://iiw.kuleuven.be/onderzoek/advise/People/PeterKarsmakers>)
- Toni Heittola (<toni.heittola@tut.fi>, <http://www.cs.tut.fi/~heittolt/>, <https://github.com/toni-heittola>)


Getting started
===============


1. Clone repository from [Github](https://github.com/DCASE-REPO/dcase2018_baseline). The baseline code for Task 5 is available under subdirectory `task5`.
2. Install requirements with command: ``pip install -r requirements.txt``. 
3. Run the application with default settings: ``python task5.py``
4. The code is mainly using the [DCASE util](https://dcase-repo.github.io/dcase_util/) library. It is advised to read the manual.


**Note:** The baseline has been tested on CentOS Linux 7.4 and Windows 7 using Python 3.6 and tensorflow 1.4. 

Baseline system description
==================

This is the baseline system for the [Task 5 of the DCASE2018 challenge](http://dcase.community/challenge2018/task-monitoring-domestic-activities). The baseline system is intended to lower the hurdle to participate the DCASE challenge(s). It provides an entry-level approach which is simple but relatively close to the state of the art systems. High-end performance is left for the challenge participants to find.

Participants are allowed to build their system on top of the given baseline system. The system has all needed functionality for dataset handling, storing / accessing features and models, and evaluating the results, making the adaptation for one's needs rather easy. The baseline system is also a good starting point for entry level researchers.

### Code summary
The file `task5.py` contains the main code of the baseline system and is largely controlled by the configuration file `task5.yaml`. The code handles downloading and reading the dataset, calculating the features and models and evaluating the results. The code is commented to assist you in understanding the structure. If there are still some questions, feel free to contact us. 

    .
    ├── task5.py            	# main code
    ├── task5.yaml         		# configuration file (system state machine control, baseline parameters, ...)
    ├── task5_datagenerator.py  # data generator class
    ├── task5_utils             # some additional functions used in the baseline system code
    ├── README.md               # This file
    └── requirements.txt        # External module dependencies 

By default, the code is set to `development mode`. In development mode results are acquired in 4-fold cross-validation based fashion. This mode is used for developing your system. The code provides an option to change to `evaluation mode` which then uses the full [development dataset](https://zenodo.org/record/1217452) to train a model to be tested on the evaluation dataset. The option for `evaluation mode` is available in the configuration file (`eval_mode: True/False`) but it is not yet fully supported. Once the evaluation set is released, we'll release an extension of this baseline.


### Feature/Machine Learning parameters

During the recording campaign, data was measured simultaneously using multiple microphone arrays (nodes) each containing 4 microphones.  Hence, each domestic activity is recorded as many times as there were microphones.  The baseline system trains a single classifier model that takes a single channel as input.  Each parallel recording of a single activity is considered as a different example during training. The learner in the baseline system is based on a Neural Network architecture using convolutional and dense layers. As input, log mel-band energies are provided to the network for each microphone channel separately. In the prediction stage a single outcome is computed for each node by averaging the 4 model outcomes (posteriors) that were computed by evaluating the trained classifier model on all 4 microphones.

The baseline system parameters are as follows:

- Frame size: 40 ms (50% hop size)
- Feature matrix: 
	- 40 log mel-band energies in 501 successive frames (10 s)
- Neural Network:
	- Input data: 40x501 (each microphone channel is considered to be a separate example for the learner)
	- Architecture:
		- 1D Convolutional layer (filters: 32, kernel size: 5, stride: 1, axis: time) + Batch Normalization + ReLU activation
		- 1D Max Pooling (pool size: 5, stride: 5) + Dropout (rate: 20%)
		- 1D Convolutional layer (filters, 64, kernel size: 3, stride: 1, axis: time) + Batch Normalization + ReLU activation
		- 1D Global Max Pooling + Dropout (rate: 20%)
		- Dense layer (neurons: 64) + ReLU activation + Dropout (rate: 20%)
		- Softmax output layer (classes: 9)
	- Learning:
		- Optimizer: Adam (learning rate: 0.0001)
		- Epochs: 500
		- On each epoch, the training dataset is randomly subsampled so that the number of examples for each class match the size of the smallest class
		- Batch size: 256 * 4 channels (each channel is considered as a different example for the learner)
- Fusion: Output probabilities from the four microphones in a particular node under test are averaged to obtain the final posterior probability.
- Model selection: The performance of the model is evaluated every 10 epochs on a validation subset (30% subsampled from the training set). The model with the highest Macro-averaged F1-score is picked.

The baseline system is build on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox. The machine learning part of the code in build on [Keras (v2.1.5)](https://keras.io/) while using [TensorFlow (v1.4.0)](https://www.tensorflow.org/) as backend.

### Baseline performance
When running in development mode (`eval_mode = False`) the baseline system provides results for a 4-fold cross-validation setup. The table below shows the averaged `Macro-averaged F1-score` over these 4 folds. The F1-score is calculated for each class seperately and averaged over all classes to obtain the `Macro-averaged F1-score`. A full 10s multi-channel audio segment is considered to be one sample.

<div class="table-responsive col-md-6">
<table class="table table-striped">
    <thead>
        <tr>
            <th>Activity</th>
            <th class="col-md-3">F1-score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Absence</td>
            <td>86.14 %</td>
        </tr>
        <tr>
            <td>Cooking</td>
            <td>94.26 %</td>
        </tr>
        <tr>
            <td>Dishwashing</td>
            <td>73.74 %</td>
        </tr>
        <tr>
            <td>Eating</td>
            <td>85.47 %</td>
        </tr>
        <tr>
            <td>Other</td>
            <td>40.41 %</td>
        </tr>  
        <tr>
            <td>Social activity</td>
            <td>94.36 %</td>
        </tr>
        <tr>
            <td>Vacuum cleaning</td>
            <td>99.16 %</td>
        </tr> 
        <tr>
            <td>Watching TV</td>
            <td>99.52 %</td>
        </tr>
        <tr>
            <td>Working</td>
            <td>81.62 %</td>
        </tr>                                                                 
    </tbody>
    <tfoot>
        <tr>
            <td><strong>Macro-averaged F1-score</strong></td>
            <td><strong>83.85 %</strong></td>
        </tr>
    </tfoot>
</table>
</div>
<div class="clearfix"></div>

**Note:** The performance might not be exactly reproducible but similar results should be obtainable.

Changelog
=========
#### 1.0.1 / 2018-04-16

* Added validation split saving for additional safety
* Minor edits

#### 1.0.0 / 2018-04-16

* First public release

## License

This software is released under the terms of the [MIT License](https://github.com/DCASE-REPO/dcase2018_baseline/blob/master/LICENSE).
