import numpy as np

from src.utils import mu_sigma
from src.graphs import *


def pca(A, n):
    # Average vector of the rows
    mu, sigma = mu_sigma(A)
    # Compute eigenvalues and eigenvectors
    s, U = np.linalg.eigh(sigma)
    # Reodered eigenvectors, reordered column of matrix U
    P = U[:, ::-1][:, 0:n]

    return np.dot(P.T, A)


class PCA(Node):
    def __init__(self, n):
        self.n = n      # Number of components
        self.P = None   # Principal components vector

    def fit(self, x):
        mu, sigma = mu_sigma(x)
        s, U = np.linalg.eigh(sigma)
        self.P = U[:, ::-1][:, 0:self.n]

    def transform(self, x):
        return np.dot(self.P.T, x)

    def __str__(self):
        return f"PCA({self.n})"
