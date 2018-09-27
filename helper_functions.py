import numpy as np
from pandas import DataFrame, read_csv, read_hdf
from sklearn.metrics import mean_squared_error


def read_hdf_to_matrix(filename):
    data = read_hdf("input/" + filename)
    return data.values, data.index


def read_csv_to_matrix(filename):
    data = read_csv("input/" + filename, index_col="Id")
    return data.values, data.index


def write_to_csv_from_vector(filename, index_col, vec):
    return DataFrame(np.c_[index_col, vec]).to_csv("output/" + filename, index=False, header=["Id", "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


def root_mean_squared_error(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5
