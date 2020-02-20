# This generates test/train data with decent values then runs training on it
# then verifies the model and displays outputs

from TestDataGeneratorCSVbatch import csv_generator
from CSVDataImporter import csv_import
from BrainWaveTrainer import model_trainer
from BrainWaveDataInference import model_inference
import sys
from colors import *


def test_runner(batch_no, epoch_no, csv_data_no):
    sys.stdout.write(BOLD)
    print("This generates artificial data to train a model then tests the model, displaying prediction results...\n")
    print("Setup with Batch number: ", batch_no, " | Epoch number: ", epoch_no, " | Amount of CSV Lines: ", csv_data_no, "\n")

    sys.stdout.write(BLUE)
    print("Generating training data...\n")
    csv_generator(int(csv_data_no), "mixed", True, "train_data.csv")
    sys.stdout.write(BOLD)
    print("Done!\n")

    sys.stdout.write(BLUE)
    print("Generating testing data...\n")
    csv_generator(int(csv_data_no/10), "mixed", True, "test_data.csv")
    sys.stdout.write(BOLD)
    print("Done!\n")

    sys.stdout.write(CYAN)
    print("Importing and Processing Data...\n")
    numeric_columns, packed_train_data, packed_test_data = csv_import(batch_no)
    sys.stdout.write(BOLD)
    print("Done!\n")

    sys.stdout.write(RED)
    print("Training Model on Data...\n")
    model = model_trainer(epoch_no, numeric_columns, packed_train_data, packed_test_data)
    sys.stdout.write(BOLD)
    print("Done!\n")

    sys.stdout.write(GREEN)
    print("Running Predictions on Model...\n")
    model_inference(model, packed_test_data)
    sys.stdout.write(BOLD)
    print("Done!\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run through a test pipeline of the brain wave analyser with '
                                                 'artificially generated data')

    parser.add_argument('-e', action="store", dest="epochs", type=int, default=10,
                        help='Number of epochs to run through')
    parser.add_argument('-b', action="store", dest="batch", type=int, default=250, help='Batches per data set')
    parser.add_argument('-c', action="store", dest="csvno", type=int, default=10000, help='Amount of lines to '
                                                                                          'generate for training '
                                                                                          'data, will be divided by '
                                                                                          '10 for test data numbers')

    args = parser.parse_args()
    batch_no = args.batch
    epoch_no = args.epochs
    csv_data_no = args.csvno

    test_runner(batch_no, epoch_no, csv_data_no)

    sys.stdout.write(RESET)
