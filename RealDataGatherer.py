from CSVCombiner import combine_csvs
from RealBrainWaveDataCSV import data_reader
import sys
from time import sleep


def gatherer(data_type, file_no):
    print("Ensure EEG headband is ON and you are wearing it")

    sleep(5)

    for x in range(file_no):

        print("Relax your mind...")

        data_reader("relaxed", "input_{0}_data_relaxed_{1}.csv".format(data_type, x))

        print("Now excite your mind...\n")

        data_reader("non-relaxed", "input_{0}_data_nonrelaxed_{1}.csv".format(data_type, x))

    combine_csvs("input_{}_".format(data_type), "{}_data.csv".format(data_type))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Gather data from EEG headband')

    parser.add_argument('-type', action="store", dest="type", type=str, choices=['train', 'test'], help='Type of reading')
    parser.add_argument('-n', action="store", dest="readings", type=int, default=2, help='Number of readings to take')

    args = parser.parse_args()
    type = args.type
    readings = args.readings

    gatherer(type, readings)

    sys.exit()

