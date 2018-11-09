from biosppy.signals import tools
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation, NMF, FastICA, FactorAnalysis, \
    DictionaryLearning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, \
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import helper_functions as hf
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import scipy.interpolate as spi
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import QuantileTransformer, StandardScaler, RobustScaler, PowerTransformer, \
    FunctionTransformer
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2, f_classif, mutual_info_classif, RFECV
from sklearn.svm import SVC, LinearSVC
from biosppy import ecg
import pywt as pt

""" 
- Different length of samples
- 4 classes: 0.0, 1.0, 2.0, 3.0
- Occurences: 0: 3030, 2: 1474, 1: 443, 3: 170 --> resampling
- ECG's (x,y): (1e-3 s, 1e-6 V) 300 Hz
"""


def read_data():
    X_train, _ = hf.read_hdf_to_matrix("X_train.h5", "id")
    y_train, _ = hf.read_hdf_to_matrix("y_train.h5", "id")
    X_test, test_index = hf.read_hdf_to_matrix("X_test.h5", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def make_pipeline():
    return Pipeline(feature_transformer(), classifier())


def grid_search(X_train, y_train, iid):
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


def scorer():
    return make_scorer(f1_score, greater_is_better=True)


def f1_score_micro(y_true, y_pred, **kwargs):
    return f1_score(y_true=y_true, y_pred=y_pred, average='micro', **kwargs)


def feature_transformer():
    return


def classifier():
    svc = SVC(
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

    clf = svc
    return clf


def evaluate():
    X_train, _, y_train, _ = read_data()
    pipe = make_pipeline()
    results = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=1, scoring=scorer())
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    return


def predict():
    X_train, X_test, y_train, test_index = read_data()
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")
    return


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


def feature_selection(x, xt):
    features = 5
    fs01 = DictionaryLearning(n_components=features)
    fs02 = FactorAnalysis(n_components=features)
    fs03 = FastICA(n_components=features)
    fs04 = NMF(n_components=features)
    fs05 = LatentDirichletAllocation(n_components=features)
    fs06 = TruncatedSVD(n_components=features)
    fs07 = SelectKBest(score_func=chi2, k=features)
    fs08 = SelectKBest(score_func=f_classif, k=features)
    fs09 = SelectKBest(score_func=mutual_info_classif, k=features)
    fs10 = LinearDiscriminantAnalysis(n_components=features)
    fs11 = PCA(n_components=100, whiten=True)
    # rfecv = RFECV(fs01)
    x = fs11.fit_transform(x)
    x_out = np.c_[
        fs01.fit_transform(x),
        fs02.fit_transform(x),
        fs03.fit_transform(x),
        fs04.fit_transform(x),
        fs05.fit_transform(x),
        fs06.fit_transform(x),
        fs07.fit_transform(x),
        fs08.fit_transform(x),
        fs09.fit_transform(x),
        fs10.fit_transform(x)
    ]

    xt_out = np.c_[
        fs01.transform(xt),
        fs02.transform(xt),
        fs03.transform(xt),
        fs04.transform(xt),
        fs05.transform(xt),
        fs06.transform(xt),
        fs07.transform(xt),
        fs08.transform(xt),
        fs09.transform(xt),
        fs10.transform(xt)
    ]

    return x_out, xt_out


def test():
    print("load data")
    X_train, X_test, y_train, test_index = read_data()

    print("fft")
    fft = FunctionTransformer(
        func=lambda m: abs(np.fft.ifft(m)),
        validate=True,
        accept_sparse=False,
        check_inverse=False
    )

    si = SimpleImputer(
        missing_values=np.nan,
        strategy='constant',  # 'mean', 'median', 'most_frequent', 'constant'
        fill_value=0
    )
    X_train = si.fit_transform(X_train)
    X_test = si.transform(X_test)
    X_train = fft.fit_transform(X_train)
    X_test = fft.transform(X_test)

    x = X_train
    y = y_train
    xt = X_test
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)

    print("pca and scale")
    rs = RobustScaler()
    pt = PowerTransformer()
    ss = StandardScaler()
    ss = QuantileTransformer()

    x = ss.fit_transform(x)
    xt = ss.transform(xt)

    x, xt = feature_selection(x, xt)

    print("clf with cv=5 and score")
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

    ovr = OneVsRestClassifier(
        svc,
        n_jobs=1
    )

    ovo = OneVsOneClassifier(
        svc,
        n_jobs=1
    )

    clf = svc
    score = make_scorer(f1_score, greater_is_better=True)

    results = cross_val_score(clf, x, y, cv=5, n_jobs=2, scoring=score, pre_dispatch='2*n_jobs')
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))

    print("create output file")
    clf.fit(x, y)

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


def analysis():
    print("load data")
    X_train, X_test, y_train, test_index = read_data()
    x = X_train[5, :]
    x = x[~np.isnan(x)]

    ecg.ecg(signal=x, sampling_rate=300)

    plt.subplot(311)
    plt.plot(x, lw=0.5)

    FFT = abs(np.fft.ifft(x))
    freqs = np.fft.fftfreq(x.size, 1)
    plt.subplot(312)
    plt.plot(freqs, np.log10(FFT), 'x')
    plt.subplot(313)
    plt.plot(freqs, FFT, 'x')
    plt.show()

    return

    x = np.nan_to_num(x)
    x_new = spi.UnivariateSpline(np.arange(0, np.size(x)), x)
    plt.subplot(411)
    plt.plot(np.arange(0, np.size(x)), x)
    plt.subplot(412)
    plt.plot(np.arange(0, np.size(x)), abs(np.fft.ifft(x)))
    plt.subplot(413)
    plt.plot(np.arange(0, np.size(x_old)), x_old)
    plt.subplot(414)
    plt.plot(np.arange(0, np.size(x_old)), x_new(np.arange(0, np.size(x_old))))
    # plt.show()

    return

    X_train = np.nan_to_num(X_train)
    print("fft")
    fft = abs(np.fft.ifft(X_train))
    print(np.shape(fft))
    print("svd")
    tsvd = sk.decomposition.truncated_svd.TruncatedSVD(n_components=100)
    tsvd.fit(fft)
    print(tsvd.singular_values_)


def extract_rpeaks(signal, sampling_rate):
    if signal is None:
        raise TypeError("Please specify an input signal.")
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(
        signal=signal,
        ftype='FIR',
        band='bandpass',
        order=order,
        frequency=[3, 45],
        sampling_rate=sampling_rate
    )

    # segment
    rpeaks, = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = correct_rpeaks(signal=filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)

    return rpeaks


def main():
    print("read data")
    X_train, X_test, y_train, test_index = read_data()

    new_x = []
    print("select features")

    ss = StandardScaler()
    fica = FastICA(n_components=2)
    fft = np.copy(X_train)
    fft = np.nan_to_num(fft)
    fft = abs(np.fft.ifft(fft))
    fft = ss.fit_transform(fft)
    fft = fica.fit_transform(fft)

    ss = StandardScaler()
    fica = FastICA(n_components=2)

    X_feat = np.copy(X_train)
    X_feat = np.nan_to_num(X_feat)
    X_feat = ss.fit_transform(X_feat)
    X_feat = fica.fit_transform(X_feat)

    for row in range(0, np.size(X_train, axis=0)):
        x = X_train[row, :]
        x = x[~np.isnan(x)]
        # cA, cD = pt.dwt(x, 'db2')
        sample_ecg = ecg.ecg(x, 300, False)
        rpeaks = np.diff(sample_ecg['rpeaks'])
        new_x.append([rpeaks.mean(), rpeaks.std()])

    new_x = np.c_[X_feat, fft, new_x]
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

    gbc = GradientBoostingClassifier(
        n_estimators=100,
        max_features='auto'
    )  # 0.66

    clf = gbc

    steps = [
        #('ss', ss),
        ('clf', clf)
    ]

    pipeline = Pipeline(steps)

    results = cross_val_score(pipeline, new_x, y_train, cv=5, n_jobs=1, scoring=scorer())
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))


main()
