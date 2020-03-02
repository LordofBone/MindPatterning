from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def model_trainer(epoch_num, numeric_columns, packed_train_data, packed_test_data):

    preprocessing_layer = tf.keras.layers.DenseFeatures(numeric_columns)

    model = tf.keras.Sequential([
      preprocessing_layer,
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad=True),
        metrics=['accuracy'])

    train_data = packed_train_data
    test_data = packed_test_data

    model.fit(train_data, epochs=epoch_num)

    test_loss, test_accuracy = model.evaluate(test_data)

    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

    #model.summary()
    model.save_weights('./relaxation_model')

    return model
