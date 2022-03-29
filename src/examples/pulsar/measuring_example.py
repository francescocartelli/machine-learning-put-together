import numpy as np
import csv

from src.classifiers import *
from plotting.plotting import plot_multiple_dcf_mindcf


def split_D_L(filename):
    reader = csv.reader(open(filename, "r"), delimiter=",")
    x = list(reader)
    TD = np.array(x).astype("float")
    return TD.T[0:-1, :], np.array(TD.T[-1, :]).astype("int")


def split_TRTE(D, L, ratio, seed=0):
    nTrain = int(D.shape[1] * ratio)
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


# Examples of training and test
if __name__ == "__main__":
    D, L = split_D_L('data/train.txt')
    (DTR, LTR), (DTE, LTE) = split_TRTE(D, L, 0.9)

    g = Gaussian(DTR, LTR)

    S_list = []
    models = ["MVG", "NBG"]
    for model in models:
        g.train(model=model)
        log_l = g.log_l(DTE)
        S_list.append(log_l[1] - log_l[0])

    plot_multiple_dcf_mindcf(S_list, LTE, np.linspace(-3, 3, 21), legend=models)

    #print(f"{model} precision: {((post_p_ratio > 0) == LTE).sum() / LTE.size * 100}%")
