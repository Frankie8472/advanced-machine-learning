import numpy as np
from keras import Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from skvideo.measure import niqe, viideo_score, viideo_features, videobliinds_features, brisque_features
from skvideo.motion import globalEdgeMotion, blockMotion

import helper_functions as hf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, \
    Dropout, Conv2D, MaxPooling2D, TimeDistributed, LSTM, LeakyReLU, Average, Lambda, K, Conv3D, MaxPooling3D, \
    GlobalMaxPooling3D, BatchNormalization, GlobalAveragePooling3D
from keras.utils import to_categorical
from imgaug import augmenters as iaa

"""
- 2 classes: 0.0, 1.0
- Occurences: 0: 79, 1: 79

"""


def read_data():
    X_train = hf.import_video_data(158, "input/train/")
    X_test = hf.import_video_data(69, "input/test/")

    test_index = np.arange(69)

    y_train, _ = hf.read_csv_to_matrix("input/train_target.csv", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def read_data_padded():
    X_train = hf.import_video_data_padded(158, "input/train/")
    X_test = hf.import_video_data_padded(69, "input/test/")

    test_index = np.arange(69)

    y_train, _ = hf.read_csv_to_matrix("input/train_target.csv", "id")
    return X_train, X_test, np.squeeze(y_train), test_index


def preprocessing(X_train, X_test, y_train):
    X_train_new = []
    X_test_new = np.empty((0, 100, 100, 1))
    y_train_new = np.asarray([])
    train_frame_index = [0]
    test_frame_index = [0]

    for sample_idx in range(X_train.shape[0]):
        sample = X_train[sample_idx]
        for frame_idx in range(sample.shape[0]):
            frame = sample[frame_idx]
            X_train_new.append(frame)

    return np.asarray(X_train_new), np.asarray(X_test_new), np.asarray(y_train_new), train_frame_index, test_frame_index


def feature_extraction(X_train, X_test, y_train):
    X_train_new = []
    X_test_new = []
    y_train_new = []
    train_frame_index = [0]
    test_frame_index = [0]

    aug = iaa.Sequential([
        iaa.Affine(scale=(0.75, 1.25), rotate=(-10, 10), translate_px=(-10, 10)),
        iaa.ContrastNormalization((0.5, 1.5))
        ], random_order=True)

    for sample_idx in range(X_train.shape[0]):
        print(str(sample_idx) + "/" + str(X_train.shape[0] - 1))

        sample_X = X_train[sample_idx]
        sample_y = y_train[sample_idx]

        vectorfields = blockMotion(videodata=sample_X, method='DS', mbSize=20, p=2)
        mean = np.mean(vectorfields, axis=(0, 1, 2))
        std = np.std(vectorfields, axis=(0, 1, 2))

        X_train_new.append(np.r_[mean, std])
        y_train_new.append(sample_y)

        # make augmentation
        for i in range(10):
            aug_det = aug.to_deterministic()
            sample_aug = aug_det.augment_images(sample_X)

            vectorfields = blockMotion(videodata=sample_aug, method='DS', mbSize=20, p=2)
            mean = np.mean(vectorfields, axis=(0, 1, 2))
            std = np.std(vectorfields, axis=(0, 1, 2))

            X_train_new.append(np.r_[mean, std])
            y_train_new.append(sample_y)

    return np.asarray(X_train_new), np.asarray(X_test_new), np.asarray(y_train_new), train_frame_index, test_frame_index


def td_avg(x):
    return K.mean(x, axis=1)


def cnn3d_model():
    model = Sequential()

    model.add(Conv3D(filters=4, kernel_size=(1, 3, 3), use_bias=False, input_shape=(209, 100, 100, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=.01))

    model.add(Conv3D(filters=8, kernel_size=(1, 3, 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=.01))

    #model.add(GlobalMaxPooling3D())
    model.add(GlobalAveragePooling3D())

    model.add(Dense(units=128))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=.01))

    model.add(Dense(units=1, use_bias=False, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_model():
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=4, kernel_size=(3, 3)), input_shape=(None, 100, 100, 1)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dense(units=1024)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(Dense(units=16)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.add(Lambda(function=lambda x: K.mean(x, axis=1), output_shape=lambda shape: (shape[0],) + shape[2:]))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def rnn_model():
    model = Sequential()

    model.add(TimeDistributed(Flatten(), input_shape=(None, 100, 100, 1)))

    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=128, return_sequences=False))

    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def crnn_model():
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=4, kernel_size=(3, 3)), input_shape=(None, 100, 100, 1)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3))))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dense(units=1024)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(Dense(units=512)))
    model.add(LeakyReLU(alpha=.1))

    model.add(TimeDistributed(Dense(units=16)))
    model.add(LeakyReLU(alpha=.1))

    model.add(LSTM(units=16, return_sequences=False))

    model.add(Dense(256, activation='relu'))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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

        print("==> Initializing model")
        model = cnn3d_model()

        print("==> Evaluation")
        global num_epoch
        for i in range(1, num_epoch + 1):
            for idx in range(X_train.shape[0]):
                model.fit(np.reshape(X_train[idx], (1, np.size(X_train[idx], axis=0), 100, 100, 1)),
                                     np.asarray([y_train[idx]]), verbose=1)
                # print(str(i) + "/" + str(num_epoch) + ": " + str(idx) + " of " + str(X_train.shape[0]-1))
            print("Finished Epoch {}".format(i))

        print("==> Predicting")
        y_pred = []
        for idx in range(X_test.shape[0]):
            y_pred_batch = model.predict(
                np.reshape(X_test[idx], (1, np.size(X_test[idx], axis=0), 100, 100, 1)), verbose=0)
            y_pred.append(np.squeeze(y_pred_batch))

        print("==> Calculating and saving score")
        error = roc_auc_score(y_test, y_pred)
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
        y_pred_batch = model.predict_on_batch(np.reshape(X_test[idx], (1, np.size(X_test[idx], axis=0), 100, 100, 1)))
        y_pred.append(y_pred_batch)

    print("==> Printing solution")
    hf.write_to_csv_from_vector("solution_franz_3ep.csv", test_index, np.asarray(y_pred), "id")
    return


def test():
    print("==> Reading data")
    X_train, X_test, y_train, test_index = read_data()

    X = []
    print("==> Feature extraction")
    for idx in range(X_train.shape[0]):
        feat = blockMotion(X_train[idx], method='DS', mbSize=5, p=2)
        mean_x = []
        mean_y = []
        std_x = []
        std_y = []
        print(str(idx) + "/" + str(X_train.shape[0]-1))
        for jdx in range(feat.shape[0]):
            step_mean_x = feat[jdx, :, :, 0].mean()
            step_mean_y = feat[jdx, :, :, 1].mean()
            step_std_x = feat[jdx, :, :, 0].std()
            step_std_y = feat[jdx, :, :, 1].std()
            mean_x.append(step_mean_x)
            mean_y.append(step_mean_y)
            std_x.append(step_std_x)
            std_y.append(step_std_y)
        mean_x = np.asarray(mean_x)
        mean_y = np.asarray(mean_y)
        std_x = np.asarray(std_x)
        std_y = np.asarray(std_y)
        X.append([
            mean_x.mean(),
            mean_y.mean(),
            mean_x.std(),
            mean_y.std(),
            std_x.mean(),
            std_y.mean(),
            std_x.std(),
            std_y.std()
        ])
    X = np.asarray(X)

    clf = SVC(
        C=10.0,
        kernel='rbf',
        gamma='scale',
        shrinking=True,
        probability=True,
        class_weight='balanced',
        verbose=False,
        max_iter=-1,
        decision_function_shape='ovr'
    )

    clf = GradientBoostingClassifier(
        n_estimators=10000,
        max_features='auto'
    )

    print(np.shape(X))
    print("==> CrossValidation")
    results = cross_val_score(clf, X, y_train, cv=5, n_jobs=5, scoring=hf.scorer())
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    return


def test2():
    print("==> Reading data")
    X_train, X_test, y_train, test_index = read_data_padded()

    clf = KerasClassifier(build_fn=cnn3d_model, epochs=20, batch_size=1, verbose=1)
    print("========= CrossValidation =========")
    results = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=5, n_jobs=1, scoring=hf.scorer(), pre_dispatch='n_jobs', verbose=1)
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    return


def test3():
    print("==> Reading data")
    X_train, X_test, y_train, test_index = read_data()

    print("==> Preprocessing")
    X_train_new, X_test_new, y_train_new, train_frame_index, test_frame_index = feature_extraction(X_train, X_test, y_train)

    clf = GradientBoostingClassifier(
        n_estimators=1000,
        max_features='auto'
    )

    clf = RandomForestClassifier()

    print("==> Crossvalidation")
    results = cross_val_score(estimator=clf, X=X_train_new, y=y_train_new, cv=5, n_jobs=5, scoring=hf.scorer(),
                              pre_dispatch='n_jobs', verbose=1)
    print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
    return

n_splits = 5
num_epoch = 3

test3()
