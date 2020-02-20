# https://www.tensorflow.org/tutorials/load_data/csv
# Thanks to them
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def model_trainer(epoch_num, numeric_columns, packed_train_data, packed_test_data):

    # Setup preprocessing layer from the packed test data numeric columns
    preprocessing_layer = tf.keras.layers.DenseFeatures(numeric_columns)

    # Connect the preprocessing layer to the other layers of the n-net
    model = tf.keras.Sequential([
      preprocessing_layer,
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    # Compile the model with the above
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad=True),
        metrics=['accuracy'])

    # Prepare train and test data from the packed data sets
    train_data = packed_train_data
    test_data = packed_test_data

    # Train the model
    model.fit(train_data, epochs=epoch_num)

    # Evaluate and display test loss/accuracy
    test_loss, test_accuracy = model.evaluate(test_data)

    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

    # Once trained return the model
    return model
