#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Read data
file_csv = open("../../results/lenet5.csv").readlines()
file_csv = list(map(lambda x: x.split(","), file_csv))

# Number of test cases
n_test = 10000

# Data
y_real = np.zeros(n_test)
y_pred = np.zeros(n_test)

for i in range(1, n_test + 1):
    y_real[i - 1] = int(file_csv[i][0][:-1])

for j in range(n_test + 2, len(file_csv)):
    num_list = np.array(file_csv[j][:-1], dtype=float)
    y_pred[j - n_test - 2] = np.argmax(num_list)


print("F1 =", f1_score(y_real, y_pred, average='weighted'))
print("Acc =", accuracy_score(y_real, y_pred))
