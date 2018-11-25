import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from skvideo.measure import viideo_features
import helper_functions as hf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, MaxPooling3D, Conv3D, Activation, Flatten, Dropout, Conv2D, Conv1D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
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
    X_train = hf.import_video_data(158, "input/train/")
    #X_test = hf.import_video_data(69, "input/test/")
    X_test = 0
    global NUMBER_OF_FEATURES
    NUMBER_OF_FEATURES = np.size(X_train, axis=0)

    y_train, test_index = hf.read_csv_to_matrix("train_target.csv", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def preprocessing(X_train, X_test):
    conv = np.zeros((100, 100, 1))
    X_train_new = []
    for n in range(0, np.size(X_train, axis=0)):
        frames = np.size(X_train[n], axis=0)
        for p in range(0, frames):
            conv += X_train[n][p]
        X_train_new.append(conv/frames)

    X_test_new = X_test
    return np.asarray(X_train_new), np.asarray(X_test_new)


def cnn_model():
    model = Sequential()
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), input_shape=(22, 100, 100, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling3D((1, 2, 2)))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling3D((1, 2, 2)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling3D((1, 2, 2)))
    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling3D((1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate():
    print("========= Reading data =========")
    X_train, X_test, y_train, test_index = read_data()

    #X_train_new, _ = preprocessing(X_train, X_test)
    X_train_new = X_train
    print("========= Evaluation =========")

    model = cnn_model()
    model.fit(X_train_new, y_train, batch_size=10, epochs=10, verbose=0)
    y_pred = model.predict_proba(X_train_new, verbose=0)
    print(y_pred)
    print('ROC_AUC: ',  roc_auc_score(y_train, y_pred))
    # results = cross_val_score(clf, X_train_new, y_train, cv=5, n_jobs=5, scoring=hf.scorer())
    # print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    return


def predict():
    X_train, X_test, y_train, test_index = read_data()

    clf = KerasClassifier(build_fn=cnn_model, epochs=40, batch_size=256, verbose=1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hf.write_to_csv_from_vector("solution.csv", test_index, y_pred, "id")
    return


evaluate()
