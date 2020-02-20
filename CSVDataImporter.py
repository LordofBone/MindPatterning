# From https://www.tensorflow.org/tutorials/load_data/csv
# Thanks to them
from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd


def csv_import(batch_num):

    tf.compat.v1.enable_eager_execution()

    # This needs tidying up at some point to be less specific
    TRAIN_DATA_URL = "file:///home/pi/Patterning/train_data.csv"
    TEST_DATA_URL = "file:///home/pi/Patterning/test_data.csv"

    train_file_path = tf.keras.utils.get_file("/home/pi/Patterning/train_data.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("/home/pi/Patterning/test_data.csv", TEST_DATA_URL)

    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)

    with open("train_data.csv") as f:
        row_count = sum(1 for line in f)

    LABEL_COLUMN = 'Relaxed'
    shuffle_buffer = (row_count * 6)

    def get_dataset(file_path, **kwargs):
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

    raw_train_data = get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)

    class PackNumericFeatures(object):
      def __init__(self, names):
        self.names = names

      def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels

    NUMERIC_FEATURES = ['Alpha Relaxation', 'Beta Concentration', 'Theta Relaxation']

    packed_train_data = raw_train_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

    packed_test_data = raw_test_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

    desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])

    def normalize_numeric_data(data, mean, std):
      # Center the data
      return (data-mean)/std

    # See what you just created.
    normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]

    return numeric_columns, packed_train_data, packed_test_data
