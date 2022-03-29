import numpy as np
from scipy.linalg import eigh
from utils import colm


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


