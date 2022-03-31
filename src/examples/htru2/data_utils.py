import numpy as np
import csv


# Get a part of data and labels from the whole set
def split_d_l(filename, ratio=1):
    reader = csv.reader(open(filename, "r"), delimiter=",")
    x = list(reader)
    TD = np.array(x).astype("float")
    TD = TD[np.random.randint(TD.shape[0], size=int(TD.shape[0] * ratio)), :]
    return TD.T[0:-1, :], np.array(TD.T[-1, :]).astype("int")


# Split data and labels between training and test samples
def split_tr_te(D, L, ratio, seed=0):
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