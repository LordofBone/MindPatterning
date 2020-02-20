# This code is from here https://github.com/TomDF47/EEG-Test-Data-Generator
# Thanks to them
from random import randint
import argparse
import random


# For the purposes of this test data generation high values are going to be considered 'non-relaxed' mind state
def csv_generator(times, datagen, noisy, file_name_out):

    # Totally random data
    def generatefull():
        global main
        main = ''
        for i in range(0, 30):
            relaxed = randint(0,1)
            alphaR = randint(0,300)/100
            betaC = randint(0,300)/100
            thetaR = randint(0,300)/100
            main += '{0},{1},{2},{3}'.format(relaxed, alphaR, betaC, thetaR) + '\n'
        return main

    # Low 'relaxed' data
    def generatelow():
        global main
        main = ''
        for i in range(0, 30):
            relaxed = (1)
            alphaR = randint(80,300)/100
            betaC = randint(30,220)/100
            thetaR = randint(80,180)/100
            main += '{0},{1},{2},{3}'.format(relaxed, alphaR, betaC, thetaR) + '\n'
        return main

    # Middle 'noisy' data
    def generatemid():
        global main
        main = ''
        for i in range(0, 30):
            relaxed = randint(0,1)
            alphaR = randint(60,240)/100
            betaC = randint(60,240)/100
            thetaR = randint(60,240)/100
            main += '{0},{1},{2},{3}'.format(relaxed, alphaR, betaC, thetaR) + '\n'
        return main

    # High 'non-relaxed' data
    def generatehigh():
        global main
        main = ''
        for i in range(0, 30):
            relaxed = (0)
            alphaR = randint(30,220)/100
            betaC = randint(80,250)/100
            thetaR = randint(20,120)/100
            main += '{0},{1},{2},{3}'.format(relaxed, alphaR, betaC, thetaR) + '\n'
        return main

    # Function for checking existing file names
    def findemptyfile(type):
        try:
            found = False
            index = 0
            while found == False:
                fileName = type + str(index) + ".csv"
                file = open(fileName, 'r')
                if file.read() == "":
                    found = True
                index = index + 1
        except IOError:
            found = True
        return type + str(index) + ".csv"

    # This is where the data can have noisy data included (middling data with random relaxed/non-relaxed
    # to emulate some inconsitencies that may exist in real data, also fully random data can be made
    # and a file name can be passed in to create test_data.csv etc without having to manually rename,
    # if no file name is passed in it will do a standard default file name that shouldn't overwrite anything else
    final = 'Relaxed,Alpha Relaxation,Beta Concentration,Theta Relaxation' + '\n'
    for i in range(0, times):
        if datagen == ("random"):
            generatefull()
            fileName_prefix = ("random_")
        else:
            if datagen == ("relaxed"):
                final += generatelow()
                fileName_prefix=("relaxed_")
            if datagen == ("non-relaxed"):
                final += generatehigh()
                fileName_prefix=("not-relaxed_")
            if datagen == ("mixed"):
                final += generatelow()
                final += generatehigh()
                fileName_prefix = ("mixed_")
            if noisy == (True):
                if random.randint(0, 100) < 30:
                    final += generatemid()
                if random.randint(0, 100) < 5:
                    generatefull()
                if file_name_out == None:
                    fileName_prefix = (fileName_prefix + "noisy_")
                else:
                    fileName_prefix = (file_name_out)
    if file_name_out == None:
        fileName = findemptyfile(str(fileName_prefix + "BatchTestCSVData"))
    else:
        fileName = (file_name_out)
    file = open(fileName, 'w')
    file.write(final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test EEG data for verifying ML')

    parser.add_argument('-t', action="store", dest="times", type=int, help='Number of rows to write')
    parser.add_argument('-n', action="store", dest="noisy", type=bool, default=False,
                        help='Whether to write noise into dataset or not (Adds middling data into the set), (True/False)')
    parser.add_argument('-data', action="store", dest="datagen", type=str, default='mixed',
                        choices=['mixed', 'relaxed', 'non-relaxed', 'random'])
    parser.add_argument('-fname', action="store", dest="file_name", type=str, default=None)

    args = parser.parse_args()

    times = args.times

    datagen = args.datagen

    noisy = args.noisy

    file_name_out = args.file_name

    csv_generator(times, datagen, noisy, file_name_out)
