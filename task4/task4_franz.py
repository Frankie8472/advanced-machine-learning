import numpy as np
import helper_functions as hf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, \
    Dropout, Conv2D, MaxPooling2D, TimeDistributed, LSTM, LeakyReLU
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
    for n in range(0, np.size(X_train, axis=0)):
        frames = np.size(X_train[n], axis=0)
        train_frame_index.append(frames + train_frame_index[n])
        X_train_new = np.append(X_train_new, X_train[n], axis=0)
        if y_train[n]:
            y_train_new = np.r_[y_train_new, np.ones(frames)]
        else:
            y_train_new = np.r_[y_train_new, np.zeros(frames)]

    for n in range(0, np.size(X_test, axis=0)):
        frames = np.size(X_test[n], axis=0)
        test_frame_index.append(frames + test_frame_index[n])
        X_test_new = np.append(X_test_new, X_test[n], axis=0)
    return np.asarray(X_train_new), np.asarray(X_test_new), np.asarray(y_train_new), train_frame_index, test_frame_index


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


def cnn_model2():
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

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def crnn_model():
    model = Sequential()
    # (100, 100, 1)
    model.add(TimeDistributed(Conv2D(filters=4, kernel_size=(3, 3)), input_shape=(None, 100, 100, 1)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    # (50, 50, 2)
    model.add(TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    # (25, 25, 4)
    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    # (12, 12, 8)
    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(units=512, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True))

    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dropout(0.5))

    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dropout(0.5))

    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate():
    print("========= Reading data ===============")
    X, _, y, _ = read_data()

    print("========= One-hot encoding y ===============")
    y = to_categorical(y=y)

    print("========= Split into train and test set =================")
    skf = StratifiedKFold(n_splits=5)
    scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print("========= Initializing CRNN model ===============")
        model = crnn_model()

        print("========= Evaluation")
        num_epoch = 2
        for i in range(1, num_epoch + 1):
            for idx in range(X_train.shape[0]):
                model.train_on_batch(np.reshape(X_train[idx], (1, np.size(X_train[idx], axis=0), 100, 100, 1)),
                                     np.asarray([y_train[idx]]))
                print(str(i) + "/" + str(num_epoch) + ": " + str(idx) + " of " + str(X_train.shape[0]))
            print("Finished Epoch {}".format(i))

        print("========= Predicting")
        y_pred = []
        for idx in range(X_test.shape[0]):
            y_pred_batch = model.predict_on_batch(
                np.reshape(X_test[idx], (1, np.size(X_test[idx], axis=0), 100, 100, 1)))
            y_pred.append(y_pred_batch)
        y_pred = np.squeeze(np.argmax(y_pred, axis=1))

        print("========= Calculating and saving score")
        scores.append(roc_auc_score(np.argmax(y_test, axis=1), y_pred))

    print("Accuracy of ROC_AUC_SCORE: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    return


def predict():
    print("========= Initializing CRNN model")
    model = crnn_model()

    print("========= Reading data")
    X_train, X_test, y_train, test_index = read_data()

    print("========= One-hot encoding y")
    y_train = to_categorical(y=y_train)

    print("========= Evaluation")
    num_epoch = 1
    for i in range(1, num_epoch + 1):
        for idx in range(X_train.shape[0]):
            model.train_on_batch(np.reshape(X_train[idx], (1, np.size(X_train[idx], axis=0), 100, 100, 1)),
                                 np.asarray([y_train[idx]]))
            print(str(i) + "/" + str(num_epoch) + ": " + str(idx) + " of " + str(X_train.shape[0]))
        print("Finished Epoch {}".format(i))

    print("========= Predicting")
    y_pred = []
    for idx in range(X_test.shape[0]):
        y_pred_batch = model.predict_proba(np.reshape(X_test[idx], (1, np.size(X_test[idx], axis=0), 100, 100, 1)))
        y_pred.append(y_pred_batch[0][1])

    print("========= Printing solution")
    hf.write_to_csv_from_vector("solution_franz_1.csv", test_index, np.asarray(y_pred), "id")
    return


evaluate()
