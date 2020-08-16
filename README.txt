Main program: sound_recognition.py
There are three variables can be specified using command line argument.
1. machine: 'l' for using processed data directly. 'r' for using data from the remote server (take 20 minutes on preprocessing data).
2. model_type: 'naive' for running the first CNN model. 'optimized' for running the improved CNN model.
3. train_new: 'y' for trainng new model, 'n' for using saved model.

For example, if user want to run the program using processed data directly, on the naive model, train new model: Python sound_recognition.py l naive y

Tool program: process_data.py, visualization.py
Those two programs can be run individually.

Run process_data.py individually will read the data from '/home/fac/cmh/tfrecord/', preprocess it for size reducing and save the data as .npy file.

Run visualization.py individually will create visualization of 1D waveform samples for each class and save them to figure/1D_waveform_sample

dir figure stores saved figure
dir model stores saved model