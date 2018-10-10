import helper_functions as hf
import numpy as np
from sklearn.metrics import r2_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.linear_model import RidgeCV, ElasticNetCV, LarsCV, LassoLarsCV, LogisticRegressionCV, LassoCV


# Feature analysis
# Add missing data
def add_missing_data(incomplete_data_train, incomplete_data_test):
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")  # "mean", "median", "most_frequent"
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


# PCA with 0.97
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
    cods = make_scorer(r2_score)
    alphas = np.geomspace(1e-10, 1e5, 16)

    reg0 = RidgeCV(
        alphas=alphas,
        fit_intercept=True,
        normalize=False,
        cv=10,
        scoring=cods
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

    reg = reg1

    reg.fit(X_train, y_train)
    print(reg.score(X_train, y_train))
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
