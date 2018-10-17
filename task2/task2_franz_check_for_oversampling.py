# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
import collections as col
from helper_functions import read_csv_to_matrix


# Get, split and transform train dataset
y_train, index_train = read_csv_to_matrix("y_train.csv", "id")

ctr = col.Counter(y_train[:, 0])
occurrences = list(ctr.values())
print("=======================================================================")
print(ctr)
print("=======================================================================")
if np.std(occurrences) < 0.2*np.mean(occurrences):
    print("Features DO NOT require re-sampling")
else:
    print("Features DO require re-sampling")
print("=======================================================================")
