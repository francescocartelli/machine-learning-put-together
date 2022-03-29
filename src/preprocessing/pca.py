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
