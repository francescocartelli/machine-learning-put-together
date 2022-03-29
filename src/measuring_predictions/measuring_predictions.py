import numpy as np
import matplotlib.pyplot as plt


def conf_matrix(pred_labels, act_labels):
    nc = np.unique(act_labels).size
    matrix = np.zeros([nc, nc], dtype=int)
    for i in range(len(act_labels)):
        matrix[int(pred_labels[i]), int(act_labels[i])] += 1
    return matrix


def dcf(cm, P, Cfp, Cfn):
    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[0, 0] + cm[1, 0])

    return P * Cfn * FNR + (1 - P) * Cfp * FPR


def norm_dcf(cm, P, Cfp, Cfn):
    DCF = dcf(cm, P, Cfp, Cfn)

    return DCF / min(P * Cfn, (1 - P) * Cfp)


def min_dcf(S, labels, P, Cfp, Cfn):
    minDCF = np.inf
    for t in S:
        normDCF = norm_dcf(conf_matrix(S > t, labels), P, Cfp, Cfn)
        minDCF = min(minDCF, normDCF)
    return minDCF


def bayes_errors_from_priors(S, labels, logPriors):
    dcfs, min_dcfs = [], []
    for logPrior in logPriors:
        prior = 1 / (1 + np.exp(-logPrior))
        dcfs.append(norm_dcf(conf_matrix(S > 0, labels), prior, 1, 1))
        min_dcfs.append(min_dcf(S, labels, prior, 1, 1))
    return dcfs, min_dcfs
