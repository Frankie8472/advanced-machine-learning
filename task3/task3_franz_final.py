from sklearn.decomposition import FastICA
from sklearn.ensemble import GradientBoostingClassifier

import helper_functions as hf
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from biosppy import ecg
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop
import pywt as pt

""" 
- Different length of samples
- 4 classes: 0.0, 1.0, 2.0, 3.0
- Occurences: 0: 3030, 2: 1474, 1: 443, 3: 170 --> resampling
- ECG's (x,y): (1e-3 s, 1e-6 V) 300 Hz
"""

SAMPLING_RATE = 300
NUMBER_OF_FEATURES = 0


def read_data():
    X_train, _ = hf.read_hdf_to_matrix("X_train.h5", "id")
    y_train, _ = hf.read_hdf_to_matrix("y_train.h5", "id")
    X_test, test_index = hf.read_hdf_to_matrix("X_test.h5", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def scorer():
    return make_scorer(f1_score_micro, greater_is_better=True)


def f1_score_micro(y_true, y_pred, **kwargs):
    return f1_score(y_true=y_true, y_pred=y_pred, average='micro', **kwargs)


def preprocessing(X_train, X_test):
    X_train_new = []
    X_test_new = []
    X_train_filtered = []
    X_test_filtered = []

    scaler = StandardScaler()
    component_analysis = FastICA(n_components=20)

    print("========= Feature extraction: X_train =========")
    for row in range(0, np.size(X_train, axis=0)):
        x_ = np.copy(X_train[row, :])
        x_ = x_[~np.isnan(x_)]
        ecg_analysis = ecg.ecg(signal=x_, sampling_rate=SAMPLING_RATE, show=False)

        if np.size(ecg_analysis['heart_rate']) == 0:
            mean_hr = 0
            var_hr = 0
        else:
            mean_hr = np.mean(ecg_analysis['heart_rate'])
            var_hr = np.var(ecg_analysis['heart_rate'])

        filtered_signal = ecg_analysis['filtered']
        mean_rpeaks = np.mean(np.diff(ecg_analysis['rpeaks']))
        var_rpeaks = np.var(np.diff(ecg_analysis['rpeaks']))
        mean_rr_I = np.mean(ecg_analysis['templates'], 0)
        var_rr_I = np.var(ecg_analysis['templates'], 0)
        cA, cD = pt.dwt(mean_rr_I, 'db3')

        X_train_filtered.append(np.pad(filtered_signal, (0, 18154 - len(filtered_signal)), mode='constant'))
        X_train_new.append(np.concatenate(([mean_hr, var_hr, mean_rpeaks], var_rpeaks, mean_rr_I, var_rr_I, cA, cD)))

    if ~np.equal(X_test, 0):
        print("========= Feature extraction: X_test =========")
        for row in range(0, np.size(X_test, axis=0)):
            x_ = np.copy(X_test[row, :])
            x_ = x_[~np.isnan(x_)]
            ecg_analysis = ecg.ecg(signal=x_, sampling_rate=SAMPLING_RATE, show=False)

            if np.size(ecg_analysis['heart_rate']) == 0:
                mean_hr = 0
                var_hr = 0
            else:
                mean_hr = np.mean(ecg_analysis['heart_rate'])
                var_hr = np.var(ecg_analysis['heart_rate'])

            filtered_signal = ecg_analysis['filtered']
            mean_rpeaks = np.mean(np.diff(ecg_analysis['rpeaks']))
            var_rpeaks = np.var(np.diff(ecg_analysis['rpeaks']))
            mean_rr_I = np.mean(ecg_analysis['templates'], 0)
            var_rr_I = np.var(ecg_analysis['templates'], 0)
            cA, cD = pt.dwt(mean_rr_I, 'db3')

            X_test_filtered.append(np.pad(filtered_signal, (0, 18154 - len(filtered_signal)), mode='constant'))
            X_test_new.append(np.concatenate(([mean_hr, var_hr, mean_rpeaks], var_rpeaks, mean_rr_I, var_rr_I, cA, cD)))

    fis = np.copy(X_train_filtered)
    fis = scaler.fit_transform(fis)
    fis = component_analysis.fit_transform(fis)

    X_train_new = np.append(X_train_new, fis, axis=1)

    if ~np.equal(X_test, 0):
        fis = np.copy(X_test_filtered)
        fis = scaler.transform(fis)
        fis = component_analysis.transform(fis)
        X_test_new = np.c_[X_test_new, fis]

    NUMBER_OF_FEATURES = np.size(X_train_new, axis=1)
    print(NUMBER_OF_FEATURES)
    print("========= End of feature extraction =========")
    return np.asarray(X_train_new), np.asarray(X_test_new)


def MLPNN_model():
    size = 512
    model = Sequential()
    model.add(Dense(size, input_dim=NUMBER_OF_FEATURES, activation='relu'))
    model.add(Dense(size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def evaluate():
    X_train, X_test, y_train, _ = read_data()
    #X_train = X_train[0:10, :]
    #y_train = y_train[0:10]
    X_train_new, _ = preprocessing(X_train, 0)

    clf = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        shrinking=True,
        probability=True,
        class_weight='balanced',
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr'
    )

    clf = KerasClassifier(build_fn=MLPNN_model, epochs=40, batch_size=256, verbose=1)

    clf = GradientBoostingClassifier(
        n_estimators=10000,
        max_features='auto'
    )  # 0.66

    print(np.shape(X_train_new))
    print("========= CrossValidation =========")
    results = cross_val_score(clf, X_train_new, y_train, cv=5, n_jobs=1, scoring=scorer())
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    return


def predict():
    X_train, X_test, y_train, test_index = read_data()
    X_train_new, X_test_new = preprocessing(X_train, X_test)

    clf = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        shrinking=True,
        probability=True,
        class_weight='balanced',
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr'
    )

    clf = KerasClassifier(build_fn=MLPNN_model, epochs=40, batch_size=256, verbose=1)

    clf.fit(X_train_new, y_train)
    y_pred = clf.predict(X_test_new)
    hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")
    return


evaluate()
