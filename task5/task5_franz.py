import numpy as np
from sklearn.decomposition import PCA

import helper_functions as hf
from scipy.stats import kurtosis, skew
from pywt import dwt
from biosppy.signals.eeg import eeg
from sklearn_pandas import cross_val_score
from sklearn.svm import SVC


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

#Attribute Information:

#1. variance of Wavelet Transformed image (continuous)

#2. skewness of Wavelet Transformed image (continuous)

#3. curtosis of Wavelet Transformed image (continuous)

#4. entropy of image (continuous)

def read_data():
    X_train_eeg1, _ = hf.read_csv_to_matrix("input/train_eeg1.csv", "Id")
    X_train_eeg2, _ = hf.read_csv_to_matrix("input/train_eeg2.csv", "Id")
    X_train_emg, _ = hf.read_csv_to_matrix("input/train_emg.csv", "Id")
    X_test_eeg1, _ = hf.read_csv_to_matrix("input/test_eeg1.csv", "Id")
    X_test_eeg2, _ = hf.read_csv_to_matrix("input/test_eeg2.csv", "Id")
    X_test_emg, _ = hf.read_csv_to_matrix("input/test_emg.csv", "Id")
    y_train, _ = hf.read_csv_to_matrix("input/train_labels.csv", "Id")
    _, test_index = hf.read_csv_to_matrix("input/sample.csv", "Id")

    return X_train_eeg1, X_train_eeg2, X_train_emg, X_test_eeg1, X_test_eeg2, X_test_emg, np.squeeze(y_train), test_index


def feature_extraction(w):
    return np.mean(w), np.std(w), kurtosis(w), skew(w), hf.mav(w), hf.rms(w)


def eeg_feature_extraction(x):
    x_new = []

    for idx in range(x.shape[0]):
        analysis = eeg(signal=x[idx].reshape(-1, x.shape[1]).transpose(), sampling_rate=128, show=False)
        v1 = analysis['filtered'].transpose()
        v2 = analysis['theta'].transpose()
        v3 = analysis['alpha_low'].transpose()
        v4 = analysis['alpha_high'].transpose()
        v5 = analysis['beta'].transpose()
        v6 = analysis['gamma'].transpose()
        x_new.append(np.r_[v1, v2, v3, v4, v5, v6, feature_extraction(v1), feature_extraction(v2), feature_extraction(v3), feature_extraction(v4), feature_extraction(v5), feature_extraction(v6)])
    return np.asarray(x_new)


def emg_feature_extraction(x):
    x_new = []
    for idx in range(x.shape[0]):
        cA, cD = dwt(data=x[idx], wavelet='db2')
        to_append = np.r_[cA, cD, feature_extraction(cA), feature_extraction(cD)]
        x_new.append(to_append)

    return np.asarray(x_new)


def evaluate():
    print("==> Reading data")
    X_train_eeg1, X_train_eeg2, X_train_emg, X_test_eeg1, X_test_eeg2, X_test_emg, y_train, test_index = read_data()
    print("==> Train feature extraction")
    print("==> EEG1")
    X_train_eeg1_new = eeg_feature_extraction(X_train_eeg1)
    print("==> EEG2")
    X_train_eeg2_new = eeg_feature_extraction(X_train_eeg2)
    print("==> EMG")
    X_train_emg_new = emg_feature_extraction(X_train_emg)

    # Fuse extracted features
    X_train = np.c_[X_train_eeg1_new, X_train_eeg2_new, X_train_emg_new]

    print("==> PCA")
    pca = PCA(
        n_components=10,
        whiten=True
    )

    X_train = pca.fit_transform(X_train)

    print("==> Initialize classifier")
    clf = SVC(
        C=1.0,
        kernel='rbf',
        shrinking=True,
        probability=False,
        class_weight='balanced'
    )

    print("==> 3 fold crossvalidation")
    scores = cross_val_score(
        estimator=clf,
        X=X_train,
        y=y_train,
        groups=None,
        scoring=hf.scorer(),
        cv=3,
        n_jobs=3,
        verbose=0,
        fit_params=None,
        pre_dispatch='2*n_jobs',
    )

    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    return


evaluate()
