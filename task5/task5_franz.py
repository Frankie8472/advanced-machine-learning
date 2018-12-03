import numpy as np
from keras import Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from skvideo.measure import niqe, viideo_score, viideo_features, videobliinds_features, brisque_features
from skvideo.motion import globalEdgeMotion, blockMotion

import helper_functions as hf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, \
    Dropout, Conv2D, MaxPooling2D, TimeDistributed, LSTM, LeakyReLU, Average, Lambda, K, Conv3D, MaxPooling3D, \
    GlobalMaxPooling3D, BatchNormalization, GlobalAveragePooling3D
from keras.utils import to_categorical
from imgaug import augmenters as iaa

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


def read_data():
    X_train_eeg1, _ = hf.read_csv_to_matrix("input/train_eeg1.csv", "Id")
    X_train_eeg2, _ = hf.read_csv_to_matrix("input/train_eeg2.csv", "Id")
    X_train_emg, _ = hf.read_csv_to_matrix("input/train_emg.csv", "Id")
    X_test_eeg1, _ = hf.read_csv_to_matrix("input/train_eeg1.csv", "Id")
    X_test_eeg2, _ = hf.read_csv_to_matrix("input/train_eeg2.csv", "Id")
    X_test_emg, _ = hf.read_csv_to_matrix("input/train_emg.csv", "Id")
    y_train, _ = hf.read_csv_to_matrix("input/train_labels.csv", "Id")
    _, test_index = hf.read_csv_to_matrix("task5/input/sample.csv", "Id")

    return X_train_eeg1, X_train_eeg2, X_train_emg, X_test_eeg1, X_test_eeg2, X_test_emg, np.squeeze(y_train), test_index


def evaluate():
    print("==> Reading data")
    X_train_eeg1, X_train_eeg2, X_train_emg, X_test_eeg1, X_test_eeg2, X_test_emg, y_train, test_index = read_data()


evaluate()
