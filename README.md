# MindPatterning
Code for my long term 'Mind Patterning' project, will read brain EEG data from a headset and then train a TensorFlow model on the data to recognise different states of mind.

You will need a Muse headset:
https://choosemuse.com/

and a Raspberry Pi (preferrably a 4):
https://shop.pimoroni.com/collections/raspberry-pi

https://blog.adafruit.com/2018/05/28/recording-brainwaves-with-a-raspberry-pi/https://github.com/alexandrebarachant/muse-lsl
https://stackoverflow.com/questions/33684894/numpy-disutils-system-info-notfounderror-no-lapack-blas-resources-found
https://stackoverflow.com/questions/29586487/still-cant-install-scipy-due-to-missing-fortran-compiler-after-brew-install-gcc

Once setup you can run:

'python3 PipelineRunnerTest.py' to run through the Pipeline with test data.

and then:

'python3 PipelineRunnerReal.py' to run through the Pipeline with real EEG data from headset.
