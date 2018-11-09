from pandas import read_csv, read_hdf


def main():
    path = "task3/input/X_test"
    file_csv = read_csv(path + ".csv", index_col='id')
    file_csv.to_hdf(path + ".h5", "X", complevel=9, complib='zlib', fletcher32=True)
    file_h5 = read_hdf(path + ".h5")
    if file_csv.equals(file_h5):
        print("===== Conversion successful =====")
    print(file_csv)
    print(file_h5)
    print(type(file_h5.values[10, 10]))


main()
