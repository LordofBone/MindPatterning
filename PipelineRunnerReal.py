# This generates test/train data with decent values then runs training on it
# then verifies the model and displays outputs

from RealDataGatherer import gatherer
from CSVDataImporter import csv_import
from BrainWaveTrainer import model_trainer
from BrainWaveDataInference import model_inference
import sys
from colors import *


def real_runner(batch_no, epoch_no):
    sys.stdout.write(BOLD)
    print("This gets real data from an EEG headband to train a model then tests the model, displaying prediction "
          "results...\n")
    print("Setup with Batch number: ", batch_no, " | Epoch number: ", epoch_no, "\n")

    sys.stdout.write(BLUE)
    print("Getting training data...\n")

    sys.stdout.write(BOLD)
    gatherer("train", 1, 10)

    print("Done!\n")

    sys.stdout.write(BLUE)
    print("Getting testing data...\n")

    sys.stdout.write(BOLD)
    gatherer("test", 1, 2)

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

    sys.exit()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run through a pipeline of the brain wave analyser with real data '
                                                 'from headband')

    parser.add_argument('-e', action="store", dest="epochs", type=int, default=10,
                        help='Number of epochs to run through')
    parser.add_argument('-b', action="store", dest="batch", type=int, default=250, help='Batches per data set')

    args = parser.parse_args()
    batch_no = args.batch
    epoch_no = args.epochs

    real_runner(batch_no, epoch_no)

    sys.stdout.write(RESET)

    sys.exit()
