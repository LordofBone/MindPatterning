from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd
import os

tf.compat.v1.enable_eager_execution()
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

LABEL_COLUMN = 'Relaxed'

NUMERIC_FEATURES = ['Alpha Relaxation', 'Beta Concentration', 'Theta Relaxation']

current_dir = os.getcwd()

# this checks for the existence of the csv files, they need to exist in order for the file paths to be set
# with keras it seems, so this will generate them if they don't exist (blank; to be overwritten later)
for x in range(2):
    try:
        TEST_DATA_URL = "file://" + current_dir + "/test_data.csv"
        test_file_path = tf.keras.utils.get_file(current_dir + "/test_data.csv", TEST_DATA_URL)
    except:
        os.system('touch test_data.csv')
        continue

for x in range(2):
    try:
        TRAIN_DATA_URL = "file://" + current_dir + "/train_data.csv"
        train_file_path = tf.keras.utils.get_file(current_dir + "/train_data.csv", TRAIN_DATA_URL)
    except:
        os.system('touch train_data.csv')
        continue


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


def get_numeric_columns(file_path):
    if file_path is None:
        file_path = test_file_path

    desc = pd.read_csv(file_path)[NUMERIC_FEATURES].describe()

    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])

    def normalize_numeric_data(data, mean, std):
        # Center the data
        return (data - mean) / std

    # See what you just created.
    normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer,
                                                      shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]

    return numeric_columns


def get_dataset(file_path, batch_num, LABEL_COLUMN, shuffle_buffer, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=batch_num, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      num_parallel_reads=5,
      na_value="?",
      num_epochs=1,
      shuffle=True,
      shuffle_buffer_size=shuffle_buffer,
      #prefetch_buffer_size=int(batch_num*3),
      ignore_errors=True,
      num_rows_for_inference=None,
      **kwargs)
  return dataset


def import_test(batch_num):


    raw_test_data = get_dataset(test_file_path, batch_num, LABEL_COLUMN, 10000)

    packed_test_data = raw_test_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

    return packed_test_data


def import_train(batch_num):

    with open("train_data.csv") as f:
        row_count = sum(1 for line in f)

    shuffle_buffer = (row_count * 6)

    raw_train_data = get_dataset(train_file_path, batch_num, LABEL_COLUMN, shuffle_buffer)

    packed_train_data = raw_train_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

    return packed_train_data


def csv_import(batch_num):

    numeric_columns = get_numeric_columns(train_file_path)

    packed_train_data = import_train(batch_num)

    packed_test_data = import_test(batch_num)

    return numeric_columns, packed_train_data, packed_test_data
