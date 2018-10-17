import helper_functions as hf
import numpy as np
from sklearn.svm import SVR
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, AllKNN
from imblearn.combine import SMOTEENN, SMOTETomek

""" 
IMPORTANT: They stupid, did use csv for large amount of data instead of hf5...

- No missing values
- 3 classes: 0.0, 1.0, 2.0
- Occurences: 1.0: 3600, 0.0: 600, 2.0: 600 --> resampling
    - 1. 1.0 reduce to 600
    - 2. 0.0 and 2.0 increase to 3000-3600
    - 3. Do 1. and 2. to 1000-3000

- test set has same imbalance:  
    1.0: 3075, 0.0: 512.5, 2.0: 512.5
    Class probability: 0.0: 0.125, 1.0: 0.75, 2.0: 0.125
"""

# Feature analysis
# Scale features between 0 and 1
def scale_features(data_train, data_test):
    mms = MinMaxScaler()
    data_train = mms.fit_transform(data_train)
    data_test = mms.transform(data_test)
    """
    qt = QuantileTransformer()
    data_train = qt.fit_transform(data_train)
    data_test = qt.transform(data_test)
    """
    return data_train, data_test


# Remove features which have low variance
def remove_features_with_low_variance(data_train, data_test):
    varth = VarianceThreshold(threshold=.9)
    varth.fit(data_train)
    data_train = varth.transform(data_train)
    data_test = varth.transform(data_test)
    return data_train, data_test


def remove_unimportant_features(data_train, data_test, y):

    skb = SelectKBest(f_classif, k=500)
    skb.fit(data_train, y)
    data_train = skb.transform(data_train)
    data_test = skb.transform(data_test)

    return data_train, data_test


def resample(data_train, y):
    ros = RandomOverSampler()
    rus = RandomUnderSampler()
    sme = SMOTEENN()
    smt = SMOTETomek()
    cnn = CondensedNearestNeighbour()
    res = rus
    data_train_resampled, y_resampled = res.fit_resample(data_train, y)
    return data_train_resampled, y_resampled


def read_data():
    X_test, test_index = hf.read_csv_to_matrix("X_test.csv", "id")
    X_train, train_index = hf.read_csv_to_matrix("X_train.csv", "id")
    y_train, train_index = hf.read_csv_to_matrix("y_train.csv", "id")
    return X_train, X_test, y_train, test_index


def predict(X_train, y_train, X_test, class_weights):
    score = make_scorer(balanced_accuracy_score, greater_is_better=True)

    skfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    svc = SVC(
        C=10.0,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-4,
        cache_size=200,
        class_weight=class_weights,
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr',
        random_state=None
    )

    results = cross_val_score(svc, X_train, y_train, cv=skfold, n_jobs=7, scoring=score)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    svc.fit(X_train, y_train)
    return svc.predict(X_test)


# Main function
def main():
    X_train, X_test, y_train, test_index = read_data()
    y_train = np.squeeze(y_train)

    print("# Train samples: "+str(X_train.shape))
    print("# Test_samples: "+str(X_test.shape))

    X_train, X_test = scale_features(X_train, X_test)
    hf.count_class_occurences(y_train)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    X_train, X_test = remove_unimportant_features(X_train, X_test, y_train)
    print(X_train.shape)
    X_train, y_train = resample(X_train, y_train)
    print("Oversampled: "+str(X_train.shape))
    hf.count_class_occurences(y_train)

    y_pred = predict(X_train, y_train, X_test, class_weights)
    hf.count_class_occurences(y_pred)
    print("Counter({1.0: 3075, 0.0: 512.5, 2.0: 512.5})  Should be!")

    hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")


main()
