import numpy as np
from utils import mu_sigma


def pca(A, n):
    # Average vector of the rows
    mu, sigma = mu_sigma(A)
    # Compute eigenvalues and eigenvectors
    s, U = np.linalg.eigh(sigma)
    # Reodered eigenvectors, reordered column of matrix U
    P = U[:, ::-1][:, 0:n]

    return np.dot(P.T, A)


class PCA:
    def __init__(self, n):
        self.n = n      # Number of components
        self.P = None   # Principal components vector

    def fit_transform(self, D):
        mu, sigma = mu_sigma(D)
        s, U = np.linalg.eigh(sigma)
        self.P = U[:, ::-1][:, 0:self.n]

        return np.dot(self.P.T, D)

    # Apply principal component to new data
    def transform(self, D):
        return np.dot(self.P.T, D)
