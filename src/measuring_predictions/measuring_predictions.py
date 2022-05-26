import numpy as np
import matplotlib.pyplot as plt


class ConfMatrix:
    def __init__(self, pred, act):
        self.cm = conf_matrix(pred, act)
        self.FNR = self.cm[0, 1] / (self.cm[0, 1] + self.cm[1, 1])
        self.FPR = self.cm[1, 0] / (self.cm[0, 0] + self.cm[1, 0])

    def fnr_fpr(self):
        return self.FNR, self.FPR


# Given predicted labels and actual label return the confusion matrix
def conf_matrix(pred_labels, act_labels):
    nc = np.unique(act_labels).size
    matrix = np.zeros([nc, nc], dtype=int)
    for i in range(len(act_labels)):
        matrix[int(pred_labels[i]), int(act_labels[i])] += 1
    return matrix


# Given a confusion matrix return the dcf
def dcf(cm, P, Cfp, Cfn):
    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[0, 0] + cm[1, 0])

    return P * Cfn * FNR + (1 - P) * Cfp * FPR


# Given a confusion matrix and a prior tercet return the normalized dcf
def norm_dcf(cm, P, Cfp, Cfn):
    DCF = dcf(cm, P, Cfp, Cfn)

    return DCF / min(P * Cfn, (1 - P) * Cfp)


# Given a confusion matrix and a prior tercet return the normalized dcf
def norm_dcf_threshold(score, labels, P, Cfp, Cfn):
    t = -np.log(P/(1-P))
    DCF = dcf(conf_matrix(score > t, labels), P, Cfp, Cfn)

    return DCF / min(P * Cfn, (1 - P) * Cfp)


# Find the min-dcf given a score vector, actual labels and prior tercet
def min_dcf(S, labels, P, Cfp, Cfn):
    minDCF = np.inf
    for t in S:
        normDCF = norm_dcf(conf_matrix(S > t, labels), P, Cfp, Cfn)
        minDCF = min(minDCF, normDCF)
    return minDCF


# Compute the dcf and min-dcf arrays for a given array of log-priors
def bayes_errors_from_priors(S, labels, logPriors):
    dcfs, min_dcfs = [], []
    for logPrior in logPriors:
        prior = 1 / (1 + np.exp(-logPrior))
        dcfs.append(norm_dcf(conf_matrix(S > 0, labels), prior, 1, 1))
        min_dcfs.append(min_dcf(S, labels, prior, 1, 1))
    return dcfs, min_dcfs


# Given a score vector and a actual labels vector compute the vector corresponding to the roc curve plot
def roc_curve_vector(S, labels, size=1000):
    roc_vector = np.zeros([size, 2])
    for i, threshold in enumerate(np.linspace(min(S), max(S), size)):
        confMatrix = conf_matrix(S > threshold, labels)
        TPR = confMatrix[0, 0] / (confMatrix[0, 0] + confMatrix[1, 0])
        FPR = confMatrix[0, 1] / (confMatrix[0, 1] + confMatrix[1, 1])
        roc_vector[i, :] = np.array([FPR, TPR])
    return roc_vector
