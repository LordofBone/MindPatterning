# From https://www.tensorflow.org/tutorials/load_data/csv
# Thanks to them
from __future__ import absolute_import, division, print_function, unicode_literals


def model_inference(model, packed_test_data):

    test_data = packed_test_data

    # Run predictions on the model against the test data
    predictions = model.predict(test_data)

    # Show the results (random batch of 50 in this case)
    for prediction, relaxed in zip(predictions[:50], list(test_data)[0][1][:50]):
      print("Predicted relaxation level: {:.2%}".format(prediction[0]),
            "| Actual outcome: ",
            ("RELAXED" if bool(relaxed) else "NOT RELAXED"), "\n")
