import numpy as np


def get_wine_data(path_train="./examples/wines/data/train.txt", path_test="./examples/wines/data/test.txt", labels=False):
    """
    Get all pulsar data and divide into labels
    """
    train_data = np.loadtxt(path_train, delimiter=",")
    test_data = np.loadtxt(path_test, delimiter=",")
    train_data, train_labels, test_data, test_labels = train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]
    if labels:
        return train_data, train_labels, test_data, test_labels
    else:
        return train_data, test_data