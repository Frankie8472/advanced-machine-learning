# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
import collections as col
from helper_functions import read_hdf


def main_1():
    # Get, split and transform train dataset
    y_train = read_hdf("task3/input/y_train.h5").values

    ctr = col.Counter(y_train[:, 0])
    occurrences = list(ctr.values())
    print("=======================================================================")
    print(ctr)
    print("=======================================================================")
    if np.std(occurrences) < 0.2 * np.mean(occurrences):
        print("Features DO NOT require re-sampling")
    else:
        print("Features DO require re-sampling")
    print("=======================================================================")


def main_2():
    # count occurences of nan
    X_train = read_hdf("task3/input/X_train.h5")
    col_name = "x18152"
    print("Total NaN: " + str(X_train.isnull().sum().sum()))
    print("Nan's in col \"" + col_name + "\": " + str(5117 - X_train[col_name].isnull().sum()))


main_1()
main_2()
