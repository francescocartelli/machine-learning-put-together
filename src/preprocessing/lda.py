import numpy as np
from scipy.linalg import eigh

from src.utils import colm
from src.graphs import *


def between_within_cov(D, L):
    classes = np.unique(L)

    mu = colm(D.mean(axis=1))

    S_B = np.zeros([D.shape[0], D.shape[0]])
    S_W = np.zeros([D.shape[0], D.shape[0]])
    for i, c in enumerate(classes):
        D_c = D[:, L == c]

        mu_c = colm(D_c.mean(1))

        # Between covariance
        mu_c_minus_mu = mu_c - mu
        S_B += D_c.shape[1] * np.dot(mu_c_minus_mu, mu_c_minus_mu.T)

        # Withing covariance
        D_minus_mu_c = D_c - mu_c
        S_W += np.dot(D_minus_mu_c, D_minus_mu_c.T)

    return S_B / L.size, S_W / L.size


def lda(D, L, m):
    S_B, S_W = between_within_cov(D, L)

    s, U = eigh(S_B, S_W)
    W = U[:, ::-1][:, 0:m]

    #UW, _, _ = np.linalg.svd(W)
    #U = UW[:, 0:m]

    return np.dot(W.T, D)


class LDA(Classifier):
    def __init__(self, m):
        self.m = m      # Number of components
        self.U = None   # Weights of lda

    def train(self, x, y):
        S_B, S_W = between_within_cov(x, y)
        s, U = eigh(S_B, S_W)
        W = U[:, ::-1][:, 0:self.m]

        UW, _, _ = np.linalg.svd(W)
        self.U = UW[:, 0:self.m]

    # Apply lda to new data
    def transform(self, x):
        return np.dot(self.U.T, x)

    def __str__(self):
        return f"LDA(m:{self.m})"

