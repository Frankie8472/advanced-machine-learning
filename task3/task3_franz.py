from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, \
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import helper_functions as hf
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, \
    FunctionTransformer
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.svm import SVC, LinearSVC
from neurokit import bio_process

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

    fft = FunctionTransformer(
        func=np.fft.ifft,
        inverse_func=np.fft.fft,
        validate=True,
        accept_sparse=False,
        pass_y=False,
        check_inverse=True
    )

    si = SimpleImputer(
        missing_values=np.nan,
        strategy='most_frequent',
        fill_value=0
    )
    X_train = si.fit_transform(X_train)
    X_test = si.transform(X_test)
    X_train = fft.fit_transform(X_train)
    X_test = fft.transform(X_test)

    y = y_train
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)

    print("fft")
    x = abs(np.fft.ifft(X_train))

    print("pca and scale")
    ss = QuantileTransformer(ignore_implicit_zeros=True)
    ss = RobustScaler()
    ss = PowerTransformer()
    ss = StandardScaler()
    pca = SelectKBest(k=100)
    pca = QuadraticDiscriminantAnalysis()
    pca = LinearDiscriminantAnalysis()
    pca = PCA(n_components=100)
    x = pca.fit_transform(x, y)
    x = ss.fit_transform(x)
    # print(x)

    print("clf with cv and score")

    mlp = MLPClassifier(  # 300 0.67
        hidden_layer_sizes=(300,),
        activation="relu",
        solver='adam',
        alpha=1.0,
        batch_size='auto',
        learning_rate="adaptive",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=2000,
        shuffle=False,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10
    )  # 0.67

    sgd = SGDClassifier(
        loss="perceptron",
        penalty='l2',
        alpha=1.0,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=False,
        verbose=0,
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
        n_iter=None
    )  # 0.59

    rid = RidgeClassifier(
        alpha=1.0,
        class_weight=None
    )  # 0.61

    lsvc = LinearSVC(
        max_iter=1000,
        class_weight=None
    )  # 0.62

    svc = SVC(
        C=3.0,
        kernel='rbf',
        degree=3,
        gamma='scale',
        shrinking=True,
        probability=True,
        class_weight=None,
        max_iter=-1
    )  # 0.6643

    etc = ExtraTreesClassifier(
        n_estimators=100,
        class_weight='balanced'
    )  # 0.61

    rfc = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_features='auto'  # None
    )  # 0.62

    dtc = DecisionTreeClassifier(
        class_weight='balanced'
    )  # 0.51

    gbc = GradientBoostingClassifier(
        n_estimators=100,
        max_features='auto'
    )  # 0.66

    abc = AdaBoostClassifier(
        base_estimator=gbc,
        n_estimators=100
    )  # 0.64

    knc = KNeighborsClassifier(
        n_neighbors=10,
        weights='uniform',  # 'uniform', 'distance'
    )  # 0.63

    rnc = RadiusNeighborsClassifier(
        radius=100.0,
        weights='uniform'
    )  # 0.59

    bc = BaggingClassifier(
        mlp,
        max_samples=0.7,
        max_features=0.7
    )  # 0.67

    estimators = [
        ('mlp', mlp),
        ('svc', svc),
        ('gbc', gbc),
        ('bc', bc)
    ]

    vc = VotingClassifier(
        estimators,
        voting='soft',
        weights=None,
        n_jobs=1,
        flatten_transform=None
    )  # soft 0.0.6838, hard: 0.6775

    clf = mlp
    score = make_scorer(f1_score, greater_is_better=True)
    skfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    results = cross_val_score(clf, x, y, cv=skfold, n_jobs=5, scoring=score)
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))

    clf.fit(x, y)

    xt = abs(np.fft.ifft(X_test))
    xt = pca.transform(xt)
    xt = ss.transform(xt)
    y_pred = clf.predict(xt)
    hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")

    return


def risky():
    print("load data")
    X_train, X_test, y_train, test_index = read_data()
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    print("ifft")
    X_test = abs(np.fft.ifft(X_test))
    X_train = abs(np.fft.ifft(X_train))

    print("clf with cv and score")

    full_labeled, full_y = np.copy(X_train), np.copy(y_train)
    full_unlabeled = np.copy(X_test)
    old_full_y_size = 0
    print(np.size(full_labeled, axis=0))
    while old_full_y_size != np.size(full_y):
        old_full_y_size = np.size(full_y)

        ss = StandardScaler()
        pca = PCA(n_components=100)

        mlp = MLPClassifier(  # 300 0.67
            hidden_layer_sizes=(300,),
            activation="relu",
            solver='adam',
            alpha=1.0,
            batch_size='auto',
            learning_rate="adaptive",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=2000,
            shuffle=False,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=10
        )  # 0.67

        svc = SVC(
            C=3.0,
            kernel='rbf',
            degree=3,
            gamma='scale',
            shrinking=True,
            probability=True,
            class_weight=None,
            max_iter=-1
        )  # 0.6643

        mlp = svc

        transformed_labeled = pca.fit_transform(full_labeled)
        transformed_unlabeled = pca.transform(full_unlabeled)
        transformed_labeled = ss.fit_transform(transformed_labeled)
        transformed_unlabeled = ss.transform(transformed_unlabeled)

        mlp.fit(transformed_labeled, full_y)
        probability_unlabeled = mlp.predict_proba(transformed_unlabeled)
        predicted_unlabeled = mlp.predict(transformed_unlabeled)

        for i in range(np.size(probability_unlabeled, 0)):
            max_prob = np.amax(probability_unlabeled[i])
            max_prob_class = predicted_unlabeled[i]
            if max_prob >= 0.9:
                full_labeled = np.r_[full_labeled, [full_unlabeled[i]]]
                full_y = np.r_[full_y, max_prob_class]
                np.delete(full_unlabeled, i, 0)

    print(np.size(full_labeled, axis=0))
    ss = StandardScaler()
    pca = PCA(n_components=100)
    svc = SVC(
        C=3.0,
        kernel='rbf',
        degree=3,
        gamma='scale',
        shrinking=True,
        probability=True,
        class_weight=None,
        max_iter=-1
    )  # 0.6643

    clf = svc

    clf.fit(full_labeled, full_y)

    xt = pca.transform(X_test)
    xt = ss.transform(xt)
    y_pred = clf.predict(xt)
    hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")


def test2():
    print("load data")
    X_train, X_test, y_train, test_index = read_data()
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    print("ifft")
    X_test = abs(np.fft.ifft(X_test))
    X_train = abs(np.fft.ifft(X_train))

    print("lda")
    print(np.size(X_train, axis=1))
    lda = LinearDiscriminantAnalysis()
    X_train = lda.fit_transform(X_train, y_train)
    print(np.size(X_train, axis=1))


test()
