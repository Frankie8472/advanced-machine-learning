from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import helper_functions as hf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.svm import SVC

""" 
- Different length of samples
- 4 classes: 0.0, 1.0, 2.0, 3.0
- Occurences: 0: 3030, 2: 1474, 1: 443, 3: 170 --> resampling
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
    X_test, test_index = hf.read_hdf_to_matrix("X_test.h5", "id")
    X_train, train_index = hf.read_hdf_to_matrix("X_train.h5", "id")
    y_train, train_index = hf.read_hdf_to_matrix("y_train.h5", "id")
    return X_train, X_test, y_train, test_index


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
        C=100.0,
        kernel='poly',
        degree=5,
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

    # evaluate(X_train, y_train, True)

    y_pred = predict(X_train, y_train, X_test)

    # hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")


def analysis():
    X_train, X_test, y_train, test_index = read_data()
    for i in range(0, 10):
        tmp = X_train[i, :]

        plt.plot(tmp)
        plt.show()

        # plt.plot(tmp)
        # plt.show()


main()
