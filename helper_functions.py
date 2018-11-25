import os

import numpy as np
import collections as col
from pandas import DataFrame, read_csv, read_hdf
from sklearn.metrics import mean_squared_error, make_scorer, f1_score, roc_auc_score
from skvideo.io import vread


def read_hdf_to_matrix(filename, index_name):
    data = read_hdf("input/" + filename)
    return data.values, data.index


def read_csv_to_matrix(filename, index_name):
    data = read_csv("input/" + filename, index_col=index_name)
    return data.values, data.index


def import_video_data(number_of_videos, folder):
    X = []
    max_eval = 209
    max = 0
    for n in range(0, number_of_videos):
        video = vread(folder + str(n) + ".avi", outputdict={"-pix_fmt": "gray"})  # [:, :, :, 0]
        X.append(video)

    return np.asarray(X)


def import_video_data_zeroed(number_of_videos, folder):
    X = []
    max_eval = 209
    max = 0
    for n in range(0, number_of_videos):
        video = vread(folder + str(n) + ".avi", outputdict={"-pix_fmt": "gray"})  # [:, :, :, 0]
        size = np.size(video, axis=0)
        video = np.append(video, np.zeros([max_eval - size, 100, 100, 1]), axis=0)
        X.append(video)
        print(np.shape(X))

    return np.asarray(X)


def write_to_csv_from_vector(filename, index_col, vec, index_name):
    return DataFrame(np.c_[index_col, vec]).to_csv("output/" + filename, index=False, header=[index_name, "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    X = data_set[:, 1:]
    return X, y


def root_mean_squared_error(y, y_pred):
    return mean_squared_error(y, y_pred) ** 5


def scorer():
    return make_scorer(roc_auc_score, greater_is_better=True)


def f1_score_micro(y_true, y_pred, **kwargs):
    return f1_score(y_true=y_true, y_pred=y_pred, average='micro', **kwargs)


def count_class_occurences(y):
    ctr = col.Counter(y)
    print("=======================================================================")
    print(ctr)
    print("=======================================================================")
    return
