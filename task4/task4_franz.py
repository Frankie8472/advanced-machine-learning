import numpy as np
import helper_functions as hf
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from skvideo.measure import viideo_features
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
    X_test = hf.import_video_data(69, "input/test/")

    test_index = np.arange(69)

    global NUMBER_OF_FEATURES
    NUMBER_OF_FEATURES = np.size(X_train, axis=0)

    y_train, _ = hf.read_csv_to_matrix("train_target.csv", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def preprocessing(X_train, X_test, y_train):
    X_train_new = np.empty((0, 100, 100, 1))
    X_test_new = np.empty((0, 100, 100, 1))
    y_train_new = np.asarray([])
    train_frame_index = [0]
    test_frame_index = [0]
    for n in range(0, np.size(X_train, axis=0)):
        frames = np.size(X_train[n], axis=0)
        train_frame_index.append(frames+train_frame_index[n])
        X_train_new = np.append(X_train_new, X_train[n], axis=0)
        if y_train[n]:
            y_train_new = np.r_[y_train_new, np.ones(frames)]
        else:
            y_train_new = np.r_[y_train_new, np.zeros(frames)]

    for n in range(0, np.size(X_test, axis=0)):
        frames = np.size(X_test[n], axis=0)
        test_frame_index.append(frames+test_frame_index[n])
        X_test_new = np.append(X_test_new, X_test[n], axis=0)
    return np.asarray(X_train_new), np.asarray(X_test_new), np.asarray(y_train_new), train_frame_index, test_frame_index


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def cnn_model():
    model = Sequential()
    model.add(Conv2D(filters=2, kernel_size=(3, 3), input_shape=(100, 100, 1)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=4, kernel_size=(3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=8, kernel_size=(3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate():
    print("========= Reading data ===============")
    X_train, X_test, y_train, test_index = read_data()

    print("========= Preprocessing data =========")
    X_train_new, X_test_new, y_train_new, train_frame_index, test_frame_index = preprocessing(X_train, X_test, y_train)
    print(np.shape(X_train_new))
    print(np.shape(y_train_new))

    print("========= Evaluation =================")

    model = cnn_model()
    model.fit(X_train_new, y_train_new, batch_size=1, epochs=1, verbose=1)
    y_pred = model.predict(X_train_new, verbose=1)
    y_pred_new = np.asarray([])
    for n in range(1, np.size(train_frame_index)):
        y_temp = y_pred[train_frame_index[n-1]:train_frame_index[n]]
        y_pred_new = np.r_[y_pred_new, np.mean(y_temp)]
    print(y_pred_new)
    print('ROC_AUC: ',  roc_auc_score(y_train, y_pred_new))

    y_pred = model.predict(X_test_new, verbose=1)
    y_pred_new = np.asarray([])
    for n in range(1, np.size(test_frame_index)):
        y_temp = y_pred[test_frame_index[n - 1]:test_frame_index[n]]
        y_pred_new = np.r_[y_pred_new, np.mean(y_temp)]

    hf.write_to_csv_from_vector("solution_franz.csv", test_index, y_pred_new, "id")
    return


def predict():
    X_train, X_test, y_train, test_index = read_data()

    clf = KerasClassifier(build_fn=cnn_model, epochs=40, batch_size=256, verbose=1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    hf.write_to_csv_from_vector("solution.csv", test_index, y_pred, "id")
    return


evaluate()
