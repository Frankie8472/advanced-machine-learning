from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

import helper_functions as hf
import numpy as np
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.linear_model import RidgeCV, ElasticNetCV


# Feature analysis
# Add missing data
def add_missing_data(incomplete_data_train, incomplete_data_test):
    imp = SimpleImputer(missing_values=np.nan, strategy="median")  # "mean", "median", "most_frequent"
    data_train = imp.fit_transform(incomplete_data_train)
    data_test = imp.transform(incomplete_data_test)
    return data_train, data_test


# Remove features which have low variance
def remove_features_with_low_variance(data_train, data_test):
    varth = VarianceThreshold(threshold=(.8 * (1 - .8)))
    varth.fit(data_train)
    data_train = varth.transform(data_train)
    data_test = varth.transform(data_test)
    return data_train, data_test


def remove_unimportant_features(data_train, data_test, y):

    skb = SelectKBest(f_regression, k=222)
    skb.fit(data_train, y)
    data_train = skb.transform(data_train)
    data_test = skb.transform(data_test)

    return data_train, data_test


# Scale features between -1 and 1
def scale_features(data_train, data_test):
    qt = QuantileTransformer()
    data_train = qt.fit_transform(data_train)
    data_test = qt.transform(data_test)
    return data_train, data_test


def read_data():
    X_test, test_index = hf.read_csv_to_matrix("X_test.csv", "id")
    X_train, train_index = hf.read_csv_to_matrix("X_train.csv", "id")
    y_train, train_index = hf.read_csv_to_matrix("y_train.csv", "id")
    return X_train, X_test, y_train, test_index


def predict(X_train, y_train, X_test):
    score = make_scorer(r2_score, greater_is_better=True)
    alphas = np.geomspace(1e-10, 1e5, 16)

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    reg0 = RidgeCV(
        alphas=alphas,
        fit_intercept=True,
        normalize=False,
        cv=10,
        scoring=score
    )

    reg1 = ElasticNetCV(
        l1_ratio=0.1,
        eps=1e-3,
        n_alphas=10000,
        alphas=alphas,
        fit_intercept=True,
        normalize=False,
        max_iter=1000000,
        tol=1e-5,
        n_jobs=3,
        cv=10,
        positive=False
    )

    reg2 = SVR(
        kernel='poly',
        degree=3,
        gamma='scale',
        coef0=1,
        tol=1e-6,
        C=1.0,
        epsilon=0.1,
        shrinking=False,
        cache_size=2000,
        verbose=False,
        max_iter=-1
    )

    reg = reg2

    results = cross_val_score(reg, X_train, y_train, cv=kfold, n_jobs=-1, scoring=score)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    reg.fit(X_train, y_train)
    return reg.predict(X_test)


# Main function
def main():
    X_train, X_test, y_train, test_index = read_data()
    X_train, X_test = add_missing_data(X_train, X_test)
    print(X_train.shape)
    X_train, X_test = remove_features_with_low_variance(X_train, X_test)
    print(X_train.shape)
    X_train, X_test = scale_features(X_train, X_test)
    X_train, X_test = remove_unimportant_features(X_train, X_test, y_train)
    print(X_train.shape)
    y_pred = predict(X_train, y_train, X_test)
    hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")

main()
