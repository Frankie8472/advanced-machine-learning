"""
 TASK 0

 Project-site:      https://aml.ise.inf.ethz.ch/task0
 Project-group:     Beneficial Exalted Neuronal Enthusiastic Randomizer (BENDER)
 Project-members:   Franz Knobel (knobelf)
                    Nicola Rüegsegger (runicola)
                    Christian Knieling (knielinc)
"""

import helperFunctions as hf


def main():
    # Import data from csv and put into matrix
    data_train = hf.read_csv_to_matrix("test.csv")
    X_test = hf.read_csv_to_matrix("train.csv")

    X_train, y_train = hf.split_into_x_y(data_train)



main()
print("Make the plan, execute the plan, expect the plan to go off the rails... throw away the plan")
