import numpy as np


def load_iris(filename="./examples/iris/data/iris.txt"):
    samples = np.genfromtxt(filename, delimiter=',')
    data, target = samples[:, 0:-1], samples[:, -1].astype('int')

    return data.T, target


def load_iris_split(filename="./examples/iris/data/iris.txt", ratio=2/3):
    samples = np.genfromtxt(filename, delimiter=',')
    data, target = samples[:, 0:-1], samples[:, -1].astype('int')

    indices = np.random.permutation(data.shape[0])
    n_train = int(indices.size * ratio)
    train_indices, test_indixes = indices[0:n_train], indices[n_train:]

    return (data[train_indices].T, target[train_indices]), (data[test_indixes].T, target[test_indixes])


def load_iris_binary_split(filename="./examples/iris/data/iris.txt", ratio=2/3):
    samples = np.genfromtxt(filename, delimiter=',')
    samples = samples[samples[:, -1] < 2]
    data, target = samples[:, 0:-1], samples[:, -1].astype('int')

    indices = np.random.permutation(data.shape[0])
    n_train = int(indices.size * ratio)
    train_indices, test_indixes = indices[0:n_train], indices[n_train:]

    return (data[train_indices].T, target[train_indices]), (data[test_indixes].T, target[test_indixes])
