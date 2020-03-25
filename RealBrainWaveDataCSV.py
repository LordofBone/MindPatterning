# From https://github.com/alexandrebarachant/muse-lsl/blob/master/examples/neurofeedback.py

from MuseStreamBegin import start_stream
import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import threading
from time import sleep
import datetime
import sys
import argparse

# Handy little enum to make code more readable


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


run1 = threading.Thread(target=start_stream, args=())
run1.start()


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


def generateline(relaxed, alphaR, betaC, thetaR):
    global main
    main = ''
    main += '{0},{1},{2},{3}'.format(relaxed, alphaR, betaC, thetaR) + '\n'
    return main


def csvwrite(relaxed_status, final, file_name):
    if file_name == None:
        fileName = findemptyfile(str(relaxed_status + "_RecordedData"))
    else:
        fileName = file_name
    file = open(fileName, 'w')
    file.write(final)


def data_reader(mind_state, file_name, time):
    # Wait for stream to begin
    sleep(20)

    """ EXPERIMENTAL PARAMETERS """
    # Modify these to change aspects of the signal processing

    # Length of the EEG data buffer (in seconds)
    # This buffer will hold last n seconds of data and be used for calculations
    BUFFER_LENGTH = 5

    # Length of the epochs used to compute the FFT (in seconds)
    EPOCH_LENGTH = 1

    # Amount of overlap between two consecutive epochs (in seconds)
    OVERLAP_LENGTH = 0.8

    # Amount to 'shift' the start of each next consecutive epoch
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

    # Index of the channel(s) (electrodes) to be used
    # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
    INDEX_CHANNEL = [0]

    final = 'Relaxed,Alpha Relaxation,Beta Concentration,Theta Relaxation' + '\n'

    if mind_state == "relaxed":
        relaxed = (1)
    else:
        relaxed = (0)
    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    # try:
    endTime = datetime.datetime.now() + datetime.timedelta(minutes=time)
    # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
    while True:
        if datetime.datetime.now() >= endTime:
            break

        """ 3.1 ACQUIRE DATA """
        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * fs))

        # Only keep the channel we're interested in
        ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

        # Update EEG buffer with the new data
        eeg_buffer, filter_state = utils.update_buffer(
            eeg_buffer, ch_data, notch=True,
            filter_state=filter_state)

        """ 3.2 COMPUTE BAND POWERS """
        # Get newest samples from the buffer
        data_epoch = utils.get_last_data(eeg_buffer,
                                         EPOCH_LENGTH * fs)

        # Compute band powers
        band_powers = utils.compute_band_powers(data_epoch, fs)
        band_buffer, _ = utils.update_buffer(band_buffer,
                                             np.asarray([band_powers]))
        # Compute the average band powers for all epochs in buffer
        # This helps to smooth out noise
        smooth_band_powers = np.mean(band_buffer, axis=0)

        # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
        #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

        """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
        # These metrics could also be used to drive brain-computer interfaces

        # Alpha Protocol:
        # Simple redout of alpha power, divided by delta waves in order to rule out noise
        alpha_metric = smooth_band_powers[Band.Alpha] / \
            smooth_band_powers[Band.Delta]
        print('Alpha Relaxation: ', alpha_metric)
        alphaR=('Alpha Relaxation: ', alpha_metric)

        # Beta Protocol:
        # Beta waves have been used as a measure of mental activity and concentration
        # This beta over theta ratio is commonly used as neurofeedback for ADHD
        beta_metric = smooth_band_powers[Band.Beta] / \
            smooth_band_powers[Band.Theta]
        print('Beta Concentration: ', beta_metric)
        betaC=('Beta Concentration: ', beta_metric)

        # Alpha/Theta Protocol:
        # This is another popular neurofeedback metric for stress reduction
        # Higher theta over alpha is supposedly associated with reduced anxiety
        theta_metric = smooth_band_powers[Band.Theta] / \
            smooth_band_powers[Band.Alpha]
        print('Theta Relaxation: ', theta_metric)
        thetaR=('Theta Relaxation: ', theta_metric)

        final += generateline(relaxed, alpha_metric, beta_metric, theta_metric)

    csvwrite(mind_state, final, file_name)
    print(str(time) + ' minutes of data gathered, Writing CSV')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record EEG data and log to CSV')

    parser.add_argument('-r', action="store", dest="relaxed_status", type=str, choices=['relaxed', 'non-relaxed'])
    parser.add_argument('-fname', action="store", dest="file_name", type=str, default=None)
    parser.add_argument('-t', action="store", dest="time", type=int, default=5, help='Time of readings to take')

    args = parser.parse_args()

    mind_state = args.relaxed_status
    file_name = args.file_name
    time = args.time

    run1 = threading.Thread(target=start_stream, args=())
    run1.start()

    data_reader(mind_state, file_name, time)

    sys.exit()
