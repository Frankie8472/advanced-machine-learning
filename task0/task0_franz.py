"""
 TASK 0

 Project-site:      https://aml.ise.inf.ethz.ch/task0
 Project-group:     Beneficial Exalted Neuronal Enthusiastic Randomizer (BENDER)
 Project-members:   Franz Knobel (knobelf)
                    Nicola Rüegsegger (runicola)
                    Christian Knieling (knielinc)
"""
from sklearn.decomposition import PCA

import helper_functions as hf
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor
from keras import Sequential
from keras.layers import Dense, Dropout

features = 1


def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(20, input_dim=features, kernel_initializer='normal', activation='tanh'))
    #model.add(Dropout(0.5))
    model.add(Dense(20, kernel_initializer='normal', activation='tanh'))
    #model.add(Dropout(0.5))
    model.add(Dense(20, kernel_initializer='normal', activation='tanh'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, kernel_initializer='normal', activation='tanh'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='tanh'))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def get_estimator():
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=0)
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

    results = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=-1, scoring=acc)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def predict(data_labeled, X_test, test_index):
    X, y = hf.split_into_x_y(data_labeled)

    pca = PCA(n_components=features)
    ss = StandardScaler()
    estimator = get_estimator()

    transformed_X = pca.fit_transform(X)
    transformed_X = ss.fit_transform(transformed_X)
    transformed_test = pca.fit_transform(X_test)
    transformed_test = ss.transform(transformed_test)

    estimator.fit(transformed_X, y)

    y_pred = estimator.predict(transformed_test)

    # Print solution to file
    hf.write_to_csv_from_vector("sample_franz.csv", test_index, y_pred)


def go():
    # Get, split and transform train dataset
    data_train, train_index = hf.read_csv_to_matrix("train.csv")
    data_test, test_index = hf.read_csv_to_matrix("test.csv")

    # Parameter search/evaluation
    # evaluate(data_train)

    # Predict y for X_test
    predict(data_train, data_test, test_index)
