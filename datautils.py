import numpy as np
import tensorflow as tf
import re
import csv
import itertools
from collections import Counter

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


def evaluate(y_true, y_pred):
    evs = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    meae = median_absolute_error(y_true, y_pred)
    rs = r2_score(y_true, y_pred)
    print('explained_variance_score = ',evs)
    print('mean_absolute_error = ', mae)
    print('mean_squared_error = ', mse)
    print('mean_squared_log_error = ',msle)
    print('median_absolute_error = ', meae)
    print('r2 score = ',rs)


class Data_helper(object):

    def __init__(self, input_path, output_path, batch_size, p_embedding_size):
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.p_embedding_size = p_embedding_size
        self.data, self.position, self.label = self.form_data()

    def read_string_input(self):
        file = open(self.input_path, 'r')
        string = file.read()
        string = string.split()
        return string

    def read_label(self):
        file = open(self.output_path, 'r')
        data = file.read()
        data = data.split()
        y = []
        for i in range(len(data)):
            y.append(float(data[i]))
        y = np.array(y)
        return y

    def form_data(self):
        sensors = self.read_string_input()
        label = self.read_label()
        alphabet = "%()+-/0123456789=BCFHKNOPS[\]il"
        char_indices = dict((c, i+1) for i, c in enumerate(alphabet))
        print("number of sensors: " + str(len(sensors)))
        print("dictionary size: " + str(len(char_indices)))
        length = [ len(s) for s in sensors]
        max_len = np.max(length)
        data_train = np.zeros((len(sensors), max_len), dtype=np.int32)
        position = np.zeros((len(sensors), max_len), dtype=np.int32)
        for i, sensor in enumerate(sensors):
            for j, element in enumerate(sensor):
                data_train[i,j] = char_indices[element]

        for i, sensor in enumerate(sensors):
            for j, element in enumerate(sensor):
                position[i,j] = j+1 if j<self.p_embedding_size else self.p_embedding_size

        return data_train, position, label

    def batch_iter(self, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data, label, position = self.data, self.label, self.position
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/self.batch_size) + 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            shuffled_postion = position[shuffle_indices]
            shuffled_label = label[shuffle_indices]
        else:
            shuffled_data = data
            shuffled_label = label
            shuffled_postion = position
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, data_size)
            yield shuffled_data[start_index:end_index], shuffled_postion[start_index:end_index], shuffled_label[start_index:end_index]