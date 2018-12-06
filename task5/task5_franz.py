import numpy as np
import helper_functions as hf
from scipy.stats import kurtosis, skew
from biosppy.signals.emg import emg


"""
In this task we will perform sequence classification. We will categorize temporally coherent and uniformly distributed 
short sections of a long time-series. In particular, for each 4 seconds of a lengthy EEG/EMG measurement of brain 
activity recorded during sleep, we will assign one of the 3 classes corresponding to the sleep stage present within 
the evaluated epoch. 

Each row in train_{eeg1,eeg2,emg}.csv is a single epoch of the corresponding channel indexed by an id, so the first 
column contains the id. In addition to the id column, each sample has 512 values corresponding to 4x128, where 4 is the 
number of seconds per epoch, and 128 is the measurement frequency. Note also that the data contains stacked recordings 
of three subjects. Each subject has 21600 epochs (24 hours) rendering in total 3x21600=64800 epochs i.e. training data 
points. Therefore, apart from the "breaks" between subjects, neighboring epochs are temporally coherent. The file 
structure is therefore:

{1:"WAKE phase",2:"NREM phase",3:"REM phase"}

Hardbaseline: 0.921524364532

Class ocurrences: {1: 34114, 2: 27133, 3: 3553}

"""
# TODO: Feature extraction
# TODO: Class balance (with 'balance' in clf or with imbalanced-learn library
# TODO: Understanding of task :P


# 1. filter eeg between 1 and 45 hz
# 2. five signal space projection (SSP) vectors were applied


def read_data():
    X_train_eeg1, _ = hf.read_csv_to_matrix("input/train_eeg1.csv", "Id")
    X_train_eeg2, _ = hf.read_csv_to_matrix("input/train_eeg2.csv", "Id")
    X_train_emg, _ = hf.read_csv_to_matrix("input/train_emg.csv", "Id")
    X_test_eeg1, _ = hf.read_csv_to_matrix("input/train_eeg1.csv", "Id")
    X_test_eeg2, _ = hf.read_csv_to_matrix("input/train_eeg2.csv", "Id")
    X_test_emg, _ = hf.read_csv_to_matrix("input/train_emg.csv", "Id")
    y_train, _ = hf.read_csv_to_matrix("input/train_labels.csv", "Id")
    _, test_index = hf.read_csv_to_matrix("input/sample.csv", "Id")

    return X_train_eeg1, X_train_eeg2, X_train_emg, X_test_eeg1, X_test_eeg2, X_test_emg, np.squeeze(y_train), test_index


def eeg_feature_extraction(x):
    x_new = []
    for i in range(x.shape[0]):
        x_new.append([np.mean(x[i]), np.std(x[i]), kurtosis(x[i]), skew(x[i])])
    return np.asarray(x_new)


def emg_feature_extraction(x):
    return emg(x)


def evaluate():
    print("==> Reading data")
    X_train_eeg1, X_train_eeg2, X_train_emg, X_test_eeg1, X_test_eeg2, X_test_emg, y_train, test_index = read_data()

    X_train_emg_new = emg(signal=X_train_emg, sampling_rate=128, show=False)


evaluate()
