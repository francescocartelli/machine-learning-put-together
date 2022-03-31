import numpy as np
import sklearn.datasets


def split_db_2to1(D, L, seed=0):
    # 100 samples for training and 50 samples for evaluation
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    # DTR and LTR training data and labels
    # DTE and LTE evaluation data and labels
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def load_iris_split():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return split_db_2to1(D, L)


def load_iris_binary_split():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

    D = D[:, L != 0]
    L = L[L != 0]
    L[L == 2] = 0

    return split_db_2to1(D, L)


def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
