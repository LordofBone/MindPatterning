from __future__ import absolute_import, division, print_function, unicode_literals


def model_inference(model, packed_test_data):

    test_data = packed_test_data

    predictions = model.predict(test_data)

    # Show some results
    for prediction, relaxed in zip(predictions[:50], list(test_data)[0][1][:50]):
      print("Predicted relaxation level: {:.2%}".format(prediction[0]),
            "| Actual outcome: ",
            ("RELAXED" if bool(relaxed) else "NOT RELAXED"), "\n")

if __name__ == '__main__':
    from CSVDataImporter import csv_import
    import tensorflow as tf

    numeric_columns, packed_train_data, packed_test_data = csv_import(5)

    preprocessing_layer = tf.keras.layers.DenseFeatures(numeric_columns)

    model = tf.keras.Sequential([
      preprocessing_layer,
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.load_weights('./relaxation_model')

    model_inference(model, packed_test_data)