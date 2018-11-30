import numpy as np
from skvideo.motion import globalEdgeMotion, blockMotion

import helper_functions as hf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, \
    Dropout, Conv2D, MaxPooling2D, TimeDistributed, LSTM, LeakyReLU, Average, Lambda, K
from keras.utils import to_categorical



"""
- 2 classes: 0.0, 1.0
- Occurences: 0: 79, 1: 79

"""


def read_data():
    X_train = hf.import_video_data(158, "input/train/")
    X_test = hf.import_video_data(69, "input/test/")

    test_index = np.arange(69)

    y_train, _ = hf.read_csv_to_matrix("train_target.csv", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def preprocessing(X_train, X_test, y_train):
    X_train_new = np.empty((0, 100, 100, 1))
    X_test_new = np.empty((0, 100, 100, 1))
    y_train_new = np.asarray([])
    train_frame_index = [0]
    test_frame_index = [0]

    return np.asarray(X_train_new), np.asarray(X_test_new), np.asarray(y_train_new), train_frame_index, test_frame_index


def td_avg(x):
    return K.mean(x, axis=1)


def cnn_model():
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=2, kernel_size=(3, 3)), input_shape=(None, 100, 100, 1)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=4, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dense(units=1028)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(Dense(units=512)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(Dense(2, activation='sigmoid')))

    model.add(Lambda(td_avg))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def rnn_model():
    model = Sequential()

    model.add(TimeDistributed(Flatten(), input_shape=(None, 100, 100, 1)))

    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128, return_sequences=False))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def crnn_model():
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3)), input_shape=(None, 100, 100, 1)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dense(units=1028)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(Dense(units=512)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(Dense(units=10)))
    model.add(LeakyReLU(alpha=.1))

    model.add(LSTM(units=10, return_sequences=False))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate():
    print("==> Reading data")
    X, _, y, _ = read_data()

    print("==> Split into train and test set")
    global n_splits
    skf = StratifiedKFold(n_splits=n_splits)
    scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("==> One-hot encoding y")
        y_train = to_categorical(y=y_train)
        y_test = to_categorical(y=y_test)

        print("==> Initializing CRNN model")
        model = cnn_model()

        print("==> Evaluation")
        global num_epoch
        for i in range(1, num_epoch + 1):
            for idx in range(X_train.shape[0]):
                model.train_on_batch(np.reshape(X_train[idx], (1, np.size(X_train[idx], axis=0), 100, 100, 1)),
                                     np.asarray([y_train[idx]]))
                print(str(i) + "/" + str(num_epoch) + ": " + str(idx) + " of " + str(X_train.shape[0]-1))
            print("Finished Epoch {}".format(i))

        print("==> Predicting")
        y_pred = []
        for idx in range(X_test.shape[0]):
            y_pred_batch = model.predict_on_batch(
                np.reshape(X_test[idx], (1, np.size(X_test[idx], axis=0), 100, 100, 1)))
            y_pred.append(np.squeeze(y_pred_batch))
        y_pred = np.argmax(y_pred, axis=1)

        print("==> Calculating and saving score")
        error = roc_auc_score(np.argmax(y_test, axis=1), y_pred)
        scores.append(error)
        print("ROC_AUC_SCORE: " + str(error))

    scores = np.asarray(scores)
    print("Accuracy of ROC_AUC_SCORE: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    return


def predict():
    print("==> Initializing CRNN model")
    model = crnn_model()

    print("==> Reading data")
    X_train, X_test, y_train, test_index = read_data()

    print("==> One-hot encoding y")
    y_train = to_categorical(y=y_train)

    print("==> Evaluation")
    global num_epoch
    for i in range(1, num_epoch + 1):
        for idx in range(X_train.shape[0]):
            model.train_on_batch(np.reshape(X_train[idx], (1, np.size(X_train[idx], axis=0), 100, 100, 1)),
                                 np.asarray([y_train[idx]]))
            print(str(i) + "/" + str(num_epoch) + ": " + str(idx) + " of " + str(X_train.shape[0]-1))
        print("Finished Epoch {}".format(i))

    print("==> Predicting")
    y_pred = []
    for idx in range(X_test.shape[0]):
        y_pred_batch = model.predict_proba(np.reshape(X_test[idx], (1, np.size(X_test[idx], axis=0), 100, 100, 1)))
        y_pred.append(y_pred_batch[0][1])

    print("==> Printing solution")
    hf.write_to_csv_from_vector("solution_franz_1.csv", test_index, np.asarray(y_pred), "id")
    return


def test():
    print("==> Reading data")
    X_train, X_test, y_train, test_index = read_data()
    for idx in range(X_train.shape[0]):
        feat = blockMotion(X_train[idx], method='ARPS', mbSize=100, p=2)

    return


n_splits = 5
num_epoch = 4

test()
