import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import helper_functions as hf
from pywt import dwt
from biosppy.signals.eeg import eeg
from biosppy.signals.emg import find_onsets
from biosppy import plotting

from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import biosppy.signals.tools as st
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


def feature_extraction(signal):
    # ensure numpy
    signal = np.array(signal)

    # mean
    mean = np.mean(signal)

    # median
    median = np.median(signal)

    # maximum amplitude
    maxAmp = np.abs(signal - mean).max()

    # variance
    sigma2 = signal.var(ddof=1)

    # standard deviation
    sigma = signal.std(ddof=1)

    # absolute deviation
    ad = np.sum(np.abs(signal - median))

    # kurtosis
    kurt = kurtosis(signal, bias=False)

    # skweness
    skewness = skew(signal, bias=False)

    return np.r_[mean, median, maxAmp, sigma2, sigma, ad, kurt, skewness]


def eeg_feature_extraction(x):
    x_new = []

    for idx in range(x.shape[0]):
        analysis = eeg(signal=x[idx].reshape(-1, x.shape[1]).transpose(), sampling_rate=128, show=False)
        # v1 = analysis['filtered'].transpose().reshape(-1)
        v2 = analysis['theta'].transpose().reshape(-1)
        v3 = analysis['alpha_low'].transpose().reshape(-1)
        v4 = analysis['alpha_high'].transpose().reshape(-1)
        v5 = analysis['beta'].transpose().reshape(-1)
        v6 = analysis['gamma'].transpose().reshape(-1)
        x_new.append(np.r_[v2, v3, v4, v5, v6])
        if idx % 1000 == 0:
            print(str(idx) + " eeg")
    return np.asarray(x_new)


def emg_feature_extraction(x):
    sampling_rate = 128
    filter_frequency = 35
    x_new = []
    for idx in range(x.shape[0]):
        signal = x[idx]
        filtered, _, _ = st.filter_signal(signal=signal,
                                          ftype='butter',
                                          band='highpass',
                                          order=4,
                                          frequency=filter_frequency,
                                          sampling_rate=sampling_rate)

        onsets = find_onsets(signal=filtered, sampling_rate=sampling_rate)

        length = len(signal)
        T = (length - 1) / sampling_rate
        ts = np.linspace(0, T, length, endpoint=False)

        # plot
        if True:
            plotting.plot_emg(ts=ts,
                              sampling_rate=1000.,
                              raw=signal,
                              filtered=filtered,
                              processed=None,
                              onsets=onsets,
                              path=None,
                              show=True)

        nr_of_onsets = np.size(onsets)
        mean_onsets = np.mean(np.diff(onsets))
        var_onsets = np.var(np.diff(onsets))

        x_new.append(np.r_[nr_of_onsets, mean_onsets, var_onsets])
    # for idx in range(x.shape[0]):
    #     cA, cD = dwt(data=x[idx], wavelet='db2')
    #     to_append = np.r_[cA]
    #     x_new.append(to_append)
    #     if idx % 1000 == 0:
    #         print(str(idx) + " eeg")
    return np.asarray(x_new)


def evaluate():
    print("==> Reading data")
    X_train_eeg1, X_train_eeg2, X_train_emg, X_test_eeg1, X_test_eeg2, X_test_emg, y_train, test_index = read_data()

    print("==> rescaling data")
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train_eeg1_1 = ss.fit_transform(X_train_eeg1[0:21600])
    X_train_eeg1_2 = ss.fit_transform(X_train_eeg1[21600:43200])
    X_train_eeg1_3 = ss.fit_transform(X_train_eeg1[43200:64800])

    X_train_eeg1_new = np.r_[X_train_eeg1_1, X_train_eeg1_2, X_train_eeg1_3]

    X_train_eeg2_1 = ss.fit_transform(X_train_eeg2[0:21600])
    X_train_eeg2_2 = ss.fit_transform(X_train_eeg2[21600:43200])
    X_train_eeg2_3 = ss.fit_transform(X_train_eeg2[43200:64800])

    X_train_eeg2_new = np.r_[X_train_eeg2_1, X_train_eeg2_2, X_train_eeg2_3]

    X_test_eeg1_1 = ss.fit_transform(X_test_eeg1[0:21600])
    X_test_eeg1_2 = ss.fit_transform(X_test_eeg1[21600:43200])

    X_test_eeg1_new = np.r_[X_test_eeg1_1, X_test_eeg1_2]

    X_test_eeg2_1 = ss.fit_transform(X_test_eeg2[0:21600])
    X_test_eeg2_2 = ss.fit_transform(X_test_eeg2[21600:43200])

    X_test_eeg2_new = np.r_[X_test_eeg2_1, X_test_eeg2_2]

    print("==> Train feature extraction")
    print("==> EEG1")
    # X_train_eeg1_new = eeg_feature_extraction(X_train_eeg1_new)
    print("==> EEG2")
    # X_train_eeg2_new = eeg_feature_extraction(X_train_eeg2_new)
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

    clf = GradientBoostingClassifier(
        n_estimators=10000,
        max_features='auto'
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
