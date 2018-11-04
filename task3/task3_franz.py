from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import helper_functions as hf
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.svm import SVC
from neurokit import bio_process
from biosppy import ecg, storage
import biosppy as bs
import pylab as pl
from imblearn.under_sampling import RandomUnderSampler

""" 
- Different length of samples
- 4 classes: 0.0, 1.0, 2.0, 3.0
- Occurences: 0: 3030, 2: 1474, 1: 443, 3: 170 --> resampling
- ECG's (x,y): (1e-3 s, 1e-6 V) 
"""


# Feature analysis
# Scale features between 0 and 1
def scale_features(data_train, data_test):
    qt = QuantileTransformer(
        n_quantiles=1000,
        output_distribution='uniform',
        ignore_implicit_zeros=False,
        subsample=int(1e5),
        random_state=None,
        copy=True

    )
    data_train = qt.fit_transform(data_train)
    data_test = qt.transform(data_test)
    return data_train, data_test


def add_missing_data(incomplete_data_train, incomplete_data_test):
    imp = SimpleImputer(missing_values=np.nan, strategy="median")  # "mean", "median", "most_frequent"
    data_train = imp.fit_transform(incomplete_data_train)
    data_test = imp.transform(incomplete_data_test)
    return data_train, data_test


def remove_unimportant_features(data_train, data_test, y):
    ft = SelectPercentile(percentile=50)
    ft = SelectKBest(k=50)
    ft.fit(data_train, y)
    data_train = ft.transform(data_train)
    data_test = ft.transform(data_test)
    return data_train, data_test


def read_data():
    X_train, _ = hf.read_hdf_to_matrix("X_train.h5", "id")
    y_train, _ = hf.read_hdf_to_matrix("y_train.h5", "id")
    X_test, test_index = hf.read_hdf_to_matrix("X_test.h5", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def evaluate(X_train, y_train, iid):
    estimator = [
        ('qt', QuantileTransformer()),
        ('sp', SelectPercentile()),
        ('svc', SVC())
    ]

    param_grid = {
        'qt__n_quantiles': [100, 1000],
        # 'qt__output_distribution': ['uniform'],
        # 'qt__ignore_implicit_zeros': [False],
        # 'qt__subsample': [int(1e4), int(1e5), int(1e6)],
        'sp__percentile': [70, 100],
        'svc__C': [0.1, 1.0, 10],
        # 'svc__kernel': ['rbf'],
        'svc__gamma': ['scale'],
        # 'svc__shrinking': [True],
        'svc__probability': [True],
        'svc__class_weight': ['balanced'],
        # 'svc__decision_function_shape': ['ovr']

    }

    grid_search = GridSearchCV(
        estimator=Pipeline(estimator),
        param_grid=param_grid,
        scoring=make_scorer(f1_score, greater_is_better=True),
        n_jobs=-1,
        pre_dispatch='2*n_jobs',
        iid=iid,
        cv=5,
        refit=False,
        verbose=2,
        error_score='raise',
        return_train_score=False,
    )

    print(np.shape(X_train))
    print(np.shape(y_train))

    grid_search.fit(X_train, y_train)

    print("======================================================================================")
    print("iid: " + str(iid))
    print("Best score:       " + str(grid_search.best_score_))
    print("Best parameters:   ")
    print("")
    print(grid_search.best_params_)
    print("")
    print("======================================================================================")


def f1_score_multi(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def predict(X_train, y_train, X_test):
    X_train, X_test = add_missing_data(X_train, X_test)
    X_train, X_test = scale_features(X_train, X_test)
    X_train, X_test = remove_unimportant_features(X_train, X_test, y_train)

    score = make_scorer(f1_score_multi, greater_is_better=True)
    print("==== Eval ====")
    skfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    svc = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        shrinking=True,
        probability=True,
        class_weight='balanced',
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr',

    )

    results = cross_val_score(svc, X_train, y_train, cv=skfold, n_jobs=1, scoring=score)
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    svc.fit(X_train, y_train)
    return svc.predict(X_test)


# Main function
def main():
    X_train, X_test, y_train, test_index = read_data()
    y_train = np.squeeze(y_train)
    tmp = X_train[3106, ~np.isnan(X_train[3106, :])]
    tmp = X_train[3107, :]
    # temp = np.concatenate((X_train[3107,enate((X_train[3107, :], np.concatenate((X_train[3107, :], X_train[3107, :]))))))
    # print(biosppy.signals.ecg.ecg(X_train[3107]))
    # print(ecg_hrv())
    # rpeaks, = biosppy.ecg.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
    # print(rpeaks)
    #    heart_rate = discrete_to_continuous(heart_rate, heart_rate_times,
    #                                        sampling_rate)  # Interpolation using 3rd order spline
    #    ecg_df["Heart_Rate"] = heart_rate

    tmp2 = bio_process(
        ecg=tmp,
        sampling_rate=100,
        ecg_filter_type="FIR",
        ecg_filter_band="bandpass",
        ecg_filter_frequency=[0.05, 150],
        ecg_segmenter="hamilton",
        ecg_quality_model="default",
        ecg_hrv_features=["time", "frequency", "nonlinear"],
    )

    # evaluate(X_train, y_train, True)

    # y_pred = predict(X_train, y_train, X_test)

    # hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")


def plot_all():
    X_train, X_test, y_train, test_index = read_data()
    y_train = np.squeeze(y_train)
    X_train = np.nan_to_num(X_train)
    j = np.random.randint(0, np.size(X_train, axis=0) - 1)

    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_resample(X_train, y_train)

    skf = StratifiedKFold(n_splits=42, shuffle=True)
    skf.get_n_splits(X_train, y_train)
    for train_index, test_index in skf.split(X_train, y_train):
        X_t, X_test = X_train[train_index], X_train[test_index]
        y_t, y_test = y_train[train_index], y_train[test_index]

        for j in range(0, np.size(X_test, axis=0)):
            plt.grid()
            yi = y_test[j]
            tmp = X_test[j, :]

            plt.subplot(311)
            plt.plot(tmp, lw=0.5)

            FFT = abs(np.fft.fft(tmp))
            freqs = np.fft.fftfreq(tmp.size, 1)

            plt.subplot(312)
            plt.plot(freqs, 20 * np.log10(FFT), 'x')
            plt.subplot(313)
            plt.plot(freqs, FFT)

            plt.savefig("images/plot_" + str(yi) + "_" + str(j).zfill(5) + ".jpg", format='jpg', dpi=1000)
            plt.clf()

        break


def analysis():
    print("load data")
    X_train, X_test, y_train, test_index = read_data()
    X_train = np.nan_to_num(X_train)
    print("fft")
    fft = abs(np.fft.ifft(X_train))
    print(np.shape(fft))
    print("svd")
    tsvd = sk.decomposition.truncated_svd.TruncatedSVD(n_components=100)
    tsvd.fit(fft)
    print(tsvd.singular_values_)

    """
    # FFT Analysis
    signal = X_train[3107, :]
    Fs = 1000
    N = len(signal)  # number of samples
    T = (N - 1) / Fs  # duration
    ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps
    pl.plot(ts, signal, lw=0.1)
    pl.grid()
    pl.savefig('plot.pdf')
    out = ecg.ecg(
        signal=signal,
        sampling_rate=Fs,
        show=False
    )
    """


def test():
    print("load data")
    X_train, X_test, y_train, test_index = read_data()
    y = np.squeeze(y_train)
    X_train = np.nan_to_num(X_train)

    print("fft")
    fft = abs(np.fft.ifft(X_train))

    print("pca and scale")
    pca = PCA(n_components=100)
    x = pca.fit_transform(fft)
    ss = StandardScaler()
    x = ss.fit_transform(x)

    print("clf with cv and score")
    svc = SVC(
        C=1.0,
        gamma='scale',
        shrinking=True,
        probability=False,
        class_weight='balanced',
        max_iter=10000
    )

    rfc = RandomForestClassifier(
        n_estimators=100
    )

    gbc = GradientBoostingClassifier(
        n_estimators=100,

    )

    clf = gbc
    score = make_scorer(f1_score_multi, greater_is_better=True)
    skfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

#    results = cross_val_score(clf, x, y, cv=skfold, n_jobs=1, scoring=score)
#    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    """
    weights = y
    weights[weights == 0.0] = 5117/3030
    weights[weights == 1.0] = 5117/443
    weights[weights == 2.0] = 5117/1474
    weights[weights == 3.0] = 5117/170
    """
    clf.fit(x, y)#, sample_weight=weights)

    xt = np.nan_to_num(X_test)
    xt = abs(np.fft.ifft(xt))
    xt = pca.transform(xt)
    xt = ss.transform(xt)
    y_pred = clf.predict(xt)
    hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")
    return


test()
