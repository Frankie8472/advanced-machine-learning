"""
 TASK 0

 Project-site:      https://aml.ise.inf.ethz.ch/task0
 Project-group:     Beneficial Exalted Neuronal Enthusiastic Randomizer (BENDER)
 Project-members:   Franz Knobel (knobelf)
                    Nicola Rüegsegger (runicola)
                    Christian Knieling (knielinc)
"""

import helper_functions as hf
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from keras import Sequential
from keras.layers import Dense

features = 9


def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(features, input_dim=features, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_estimator():
    estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=100, verbose=0)
    return estimator


def evaluate(data_labeled):
    X, y = hf.split_into_x_y(data_labeled)

    pca = PCA(n_components=features)
    ss = StandardScaler()

    transformed_X = pca.fit_transform(X)
    transformed_X = ss.fit_transform(transformed_X)

    kf = KFold(n_splits=10, shuffle=False, random_state=42)
    acc = make_scorer(hf.mean_squared_error, greater_is_better=False)
    estimator = get_estimator()

    results = cross_val_score(estimator, transformed_X, y, scoring=acc, cv=kf, n_jobs=2)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def predict(data_labeled, X_test):
    X, y = hf.split_into_x_y(data_labeled)

    pca = PCA(n_components=features)
    ss = StandardScaler()
    estimator = get_estimator()

    transformed_X = pca.fit_transform(X)
    transformed_X = ss.fit_transform(transformed_X)
    transformed_test = pca.transform(X_test)
    transformed_test = ss.transform(transformed_test)

    estimator.fit(transformed_X, y)

    y_pred = estimator.predict(transformed_test)

    # Print solution to file
    hf.write_to_csv_from_vector("sample_franz.csv", test_index, y_pred)


if __name__ == "__main__":
    # Get, split and transform train dataset
    data_train, train_index = hf.read_csv_to_matrix("test.csv")
    data_test, test_index = hf.read_csv_to_matrix("test.csv")

    # Parameter search/evaluation
    # evaluate(data_train_labeled)
    evaluate(data_train)
