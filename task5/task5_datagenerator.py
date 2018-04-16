import numpy
import keras
import sklearn

# =====================================================================
# Class: Data generator
# =====================================================================
class DataGenerator(keras.utils.Sequence):
    def __init__(self, train_filename_list, labels, data_processing_chain=None, batches=-1, batch_size=256,  undersampling='uniform', shuffle=True):
        """Data generator

        Parameters
        ----------
        train_filename_list : list
           Dataset

        labels : numpy.array
           labels

        data_processing_chain : dcase_util.processors.ProcessingChain
           Data processing chain (e.g. Reading, Normalization, ...)

        batches : int
           Total amount of batches

        batch_size : int
           Batch size

        undersampling : str ('uniform','none')
           Uniform undersampling or none

        shuffle : bool
           shuffle epoch

        """

        # inits
        self.batch_size = batch_size
        self.batches = batches
        self.labels = labels
        self.data_processing_chain = data_processing_chain
        self.undersampling = undersampling
        self.train_filename_list = train_filename_list
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # batch size
        if self.batches < 1:
            self.batches = len(self.indexes) // self.batch_size
        return self.batches

    def __getitem__(self, index):
        # Select batch
        start = index * self.batch_size
        stop = min((index + 1) * self.batch_size, self.nr_examples) #if needed provide smaller batch (used in validation)
        training_files_tmp = [self.train_filename_list[k] for k in self.indexes[start:stop]]
        y_tmp = self.labels[self.indexes[start:stop]]

        # Generate data
        X, y = self.__data_generation(training_files_tmp,y_tmp)

        return X, y

    def on_epoch_end(self):
        # undersampling
        if self.undersampling == 'uniform':
            classes = numpy.unique(self.labels)
            nr_class = len(classes)
            class_weight = sklearn.utils.compute_class_weight('balanced', classes, self.labels)
            min_ex = numpy.argmax(class_weight)
            values_to_sample = numpy.sum(self.labels == min_ex)
            all_indexes = numpy.arange(len(self.labels))
            self.indexes = [None] * (values_to_sample * nr_class)
            for i in range(len(numpy.unique(self.labels))):
                self.indexes[i * values_to_sample: (i + 1) * values_to_sample] = numpy.random.choice(all_indexes[self.labels == i], values_to_sample, replace=False)
        else:
            self.indexes = numpy.arange(len(self.train_filename_list))
        self.nr_examples = len(self.indexes)

        # shuffle data
        if self.shuffle == True:
            numpy.random.shuffle(self.indexes)

    def __data_generation(self, training_files_tmp, y_tmp):
        # Initialization
        X = []
        y = []

        # Generate data
        for j, ID in enumerate(training_files_tmp):
            # load data
            features = self.data_processing_chain.process(filename=ID)

            # stack data
            # note: channels are considered to be seperate examples, therefor the labels are tiled.
            #       During recognition the posteriors for each channel are averaged to provide a final posterior
            if len(X) == 0:
                X = features.data  # init by first entry
                y = numpy.tile(y_tmp[j], numpy.ma.size(features.data, 0))  # init by first entry
            else:
                X = numpy.append(X, features.data, axis=0)  # stack images
                y = numpy.append(y, numpy.tile(y_tmp[j], numpy.ma.size(features.data, 0)))  # stack labels

        return X, y