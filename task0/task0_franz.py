"""
 TASK 0

 Project-site:      https://aml.ise.inf.ethz.ch/task0
 Project-group:     Beneficial Exalted Neuronal Enthusiastic Randomizer (BENDER)
 Project-members:   Franz Knobel (knobelf)
                    Nicola Rüegsegger (runicola)
                    Christian Knieling (knielinc)
"""
from sklearn.pipeline import Pipeline

import helper_functions as hf
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from keras import Sequential
from keras.layers import Dense, Dropout

features = 10


def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(features, input_dim=features, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    """model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))"""
    model.add(Dense(1, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def get_estimator():
    estimator = KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1)
    return estimator


def evaluate(data_labeled):
    X, y = hf.split_into_x_y(data_labeled)

    estimator = get_estimator()

    estimators = [
        ('standardize', StandardScaler()),
        ('mlp', estimator)
    ]

    pipeline = Pipeline(estimators)

    kfold = KFold(n_splits=10, shuffle=False, random_state=42)
    acc = make_scorer(hf.root_mean_squared_error, greater_is_better=False)

    results = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=3)   # scoring=acc,
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def predict(data_labeled, X_test):
    X, y = hf.split_into_x_y(data_labeled)

    ss = StandardScaler()
    estimator = get_estimator()

    transformed_X = ss.fit_transform(X)
    transformed_test = ss.transform(X_test)

    estimator.fit(transformed_X, y)

    y_pred = estimator.predict(transformed_test)

    # Print solution to file
    hf.write_to_csv_from_vector("sample_franz.csv", test_index, y_pred)


if __name__ == "__main__":
    # Get, split and transform train dataset
    data_train, train_index = hf.read_csv_to_matrix("train.csv")
    data_test, test_index = hf.read_csv_to_matrix("test.csv")

    # Parameter search/evaluation
    evaluate(data_train)
