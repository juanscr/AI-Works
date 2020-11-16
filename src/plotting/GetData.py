#!/usr/bin/env python3

import numpy as np

# Normalize data
def normalize(data):
    return data / np.max(np.abs(data), axis=0)

# Purge columns
def purge(column):
    index = np.where(column == -2)[0][0]
    return column[:index]

# Create data set
def create_data(name_file, index_file, sep=True):
    data_csv = open(name_file).readlines()[1:]
    data = np.zeros((len(data_csv), 7))
    k = 0
    for row in data_csv:
        data[k, :] = list(map(float, row.split(",")))
        k += 1

    data = normalize(data)

    if not sep:
        return data[:, :6], data[:, 6:]

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

    # Data creation
    train_datax = data[indexes_tr, :6]
    train_datay = data[indexes_tr, 6:]

    test_datax = data[indexes_te, :6]
    test_datay = data[indexes_te, 6:]

    val_datax = data[indexes_val, :6]
    val_datay = data[indexes_val, 6:]

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
