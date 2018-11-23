import numpy as np
import helper_functions as hf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, MaxPooling3D, Conv3D, Activation, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop


"""
- 2 classes: 0.0, 1.0
- Occurences: 0: 79, 1: 79

1. cropp to view (ev. with u-net)
2. cnn multi input (array of 3, color depth)
3. 2 output, classification -> keras
"""

NUMBER_OF_FEATURES = 0


def read_data():
    X_train = hf.import_data(158, "input/train/", "avi", True)
    X_test = hf.import_data(69, "input/test/", "avi", True)

    global NUMBER_OF_FEATURES
    NUMBER_OF_FEATURES = np.size(X_train, axis=0)

    y_train, test_index = hf.read_csv_to_matrix("input/train_target.csv", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def preprocessing(X_train, X_test):
    X_train_new = X_train
    X_test_new = X_test
    return np.asarray(X_train_new), np.asarray(X_test_new)


def cnn_model():
    global NUMBER_OF_FEATURES
    model = Sequential()
    model.add(Conv3D(input_shape=(209, 100, 100, 3), filters=32, pool_size=(1, 60, 1), strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 30, 1), strides=2, padding='valid', data_format=None))
    model.add(Conv3D(32, (1, 30, 1), strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1)))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling3D(pool_size=(1, 15, 1), strides=2, padding='valid', data_format=None))
    model.add(Conv3D(32, (1, 15, 1), strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1)))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling3D(pool_size=(1, 8, 1), strides=2, padding='valid', data_format=None))
    model.add(Conv3D(32, (1, 8, 1), strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model


def evaluate():
    X_train, X_test, y_train, test_index = read_data()
    X_train = X_train[0:100, :]
    y_train = y_train[0:100]
    X_train_new, _ = preprocessing(X_train, 0)

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

    clf = KerasClassifier(build_fn=cnn_model, epochs=40, batch_size=256, verbose=0)

    clf = GradientBoostingClassifier(
        n_estimators=1000,
        max_features='auto'
    )  # 0.66

    print(np.shape(X_train_new))
    print("========= CrossValidation =========")
    results = cross_val_score(clf, X_train_new, y_train, cv=5, n_jobs=1, scoring=hf.scorer())
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    return


def predict():
    X_train, X_test, y_train, test_index = read_data()
    X_train_new, X_test_new = preprocessing(X_train, X_test)

    clf = SVC(
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

    clf = KerasClassifier(build_fn=cnn_model, epochs=40, batch_size=256, verbose=1)

    clf = GradientBoostingClassifier(
        n_estimators=1000,
        max_features='auto'
    )  # 0.66

    clf.fit(X_train_new, y_train)
    y_pred = clf.predict(X_test_new)
    hf.write_to_csv_from_vector("output_franz.csv", test_index, y_pred, "id")
    return


def test():
    return


test()
