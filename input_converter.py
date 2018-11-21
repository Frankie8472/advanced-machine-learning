from pandas import read_csv, read_hdf, DataFrame
from skvideo.io import vread
import numpy as np

def csv_to_hdf5():
    path = "task3/input/X_test"
    file_csv = read_csv(path + ".csv", index_col='id')
    file_csv.to_hdf(path + ".h5", "X", complevel=9, complib='zlib', fletcher32=True)
    file_h5 = read_hdf(path + ".h5")
    if file_csv.equals(file_h5):
        print("===== Conversion successful =====")
    print(file_csv)
    print(file_h5)
    print(type(file_h5.values[10, 10]))


def video_to_hdf5(number_of_videos, folder, format, starts_with_zero):
    q = 0
    if not starts_with_zero: q = 1
    X = np.asarray([])
    max_eval = 209
    max = 0
    for n in range(0+q, number_of_videos+q):
        sample = vread(folder+str(n)+"."+format)
        number_of_frames = np.ma.size(sample, axis=0)
        frames_to_add = max_eval - number_of_frames
        slice_to_add = np.zeros(shape=(frames_to_add, 100, 100, 3))
        if number_of_frames > max:
            max = np.ma.size(sample, axis=0)
        sample = np.concatenate((sample, slice_to_add), axis=0)
        np.append(X, sample)
    #print(max)

    return


def main():
    video_to_hdf5(158, "task4/input/train/", "avi", True)
    video_to_hdf5(69, "task4/input/test/", "avi", True)
    return


main()
