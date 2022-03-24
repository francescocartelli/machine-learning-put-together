import numpy as np


def pca(A, n):
    # Average vector of the rows
    mu = A.mean(1)
    # Centered A matrix
    AC = A - mu.reshape(mu.size, 1)
    # Covariance matrix
    C = (1 / A.shape[1]) * np.dot(AC, AC.T)
    # Compute eigenvalues and eigenvectors
    s, U = np.linalg.eigh(C)
    # Reodered eigenvectors, reordered column of matrix U
    P = U[:, ::-1][:, 0:n]

    return np.dot(P.T, A)
