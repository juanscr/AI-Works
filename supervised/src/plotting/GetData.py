#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

rc('text', usetex=True)
save = lambda x: plt.savefig("../../article/figs/" + x, bbox_inches='tight')

# Normalize data
def normalize(data):
    return data / np.max(np.abs(data), axis=0)

# Purge columns
def purge(column):
    index = np.where(column == -2)[0][0]
    return column[:index]

# Calculate sensitivity and specificity
def get_sensitivity_specifity(out_clas, real_clas, threshold=0.2):
    # Sensitivity
    tp = 0
    fn = 0

    # Specificity
    tn = 0
    fp = 0

    # Calculate each one
    for i in range(out_clas.size):
        if 0.5 - threshold <= out_clas[i] <= 0.5 + threshold:
            continue

        if out_clas[i] < 0.5 - threshold:
            cond = int(real_clas[i] == 0)
            tn += cond
            fn += 1 - cond
        elif out_clas[i] > 0.5 + threshold:
            cond = int(real_clas[i] == 1)
            tp += cond
            fp += 1 - cond
    try:
        sens = tp / (tp + fn)
    except ZeroDivisionError:
        sens = "Error"

    try:
        spec = tn / (tn + fp)
    except ZeroDivisionError:
        spec = "Error"

    return sens, spec

# Create file for classifiers
def create_sens_spec(outs, name_file, sep_info, data_file, index_file):
    file0 = open(name_file, "w")

    indexes_tr, indexes_te, indexes_val = get_index(index_file)
    _, datay = create_data(data_file, index_file, sep=False)

    for i in range(len(outs)):
        out = outs[i]
        for key in sep_info[i]:
            file0.write(str(key) + "," + str(sep_info[i][key]) + "\n")

        sens_tr, spec_tr = get_sensitivity_specifity(out[indexes_tr],
                                                     datay[indexes_tr, 0])
        sens_te, spec_te = get_sensitivity_specifity(out[indexes_te],
                                                     datay[indexes_te, 0])
        sens_val, spec_val = get_sensitivity_specifity(out[indexes_val],
                                                       datay[indexes_val, 0])

        file0.write("Training,"+ str(sens_tr) + "," + str(spec_tr)+ "\n")
        file0.write("Testing,"+ str(sens_te) + "," + str(spec_te) + "\n")
        file0.write("Validation,"+ str(sens_val) + "," + str(spec_val) + "\n")
    file0.close()

# Get indexes for datasets
def get_index(index_file):
    # Separate data
    indexes_csv = open(index_file).readlines()[1:]
    indexes = np.zeros((len(indexes_csv), 3), dtype=int)
    k = 0
    for row in indexes_csv:
        indexes[k, :] = list(map(lambda x: int(float(x)), row.split(",")))
        k += 1
    indexes = indexes - 1

    # Indexes
    indexes_tr = indexes[:, 0]
    indexes_te = purge(indexes[:, 1])
    indexes_val = purge(indexes[:, 2])

    return indexes_tr, indexes_te, indexes_val

# Create data set
def create_data(name_file, index_file, sep=True):
    data_csv = open(name_file).readlines()[1:]
    data = []
    for row in data_csv:
        data.append(row.split(","))

    data = normalize(np.array(data, dtype=float))

    if not sep:
        return data[:, :-1], data[:, -1:]

    indexes_tr, indexes_te, indexes_val = get_index(index_file)

    # Data creation
    train_datax = data[indexes_tr, :-1]
    train_datay = data[indexes_tr, -1:]

    test_datax = data[indexes_te, :-1]
    test_datay = data[indexes_te, -1:]

    val_datax = data[indexes_val, :-1]
    val_datay = data[indexes_val, -1:]

    return (train_datax, train_datay), (test_datax, test_datay),\
           (val_datax, val_datay)

# Create meshgrid
def create_meshgrid(name_file):
    mesh_csv = open(name_file).readlines()[1:]
    mesh = np.zeros((len(mesh_csv), 6))
    k = 0
    for row in mesh_csv:
        mesh[k, :] = list(map(float, row.split(",")))
        k += 1
    return mesh

def plot_embedded_data(embedded_data_file):
    datax, _ = create_data(embedded_data_file, "../data/indexes.csv", sep=False)
    plt.scatter(datax[:, 0], datax[:, 1], color='k')
    save("embedded-data.pdf")
