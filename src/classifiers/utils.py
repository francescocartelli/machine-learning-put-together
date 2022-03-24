import numpy as np


# Return column matrix
def colm(v):
    return v.reshape(v.size, 1)


# Return row matrix
def rowm(v):
    return v.reshape(1, v.size)


# Return mean vector and covariance matrix
def mu_sigma(A):
    mu = A.mean(1).reshape(A.shape[0], 1)
    AC = A - mu
    sigma = np.dot(AC, AC.T) / A.shape[1]
    return mu, sigma


# Sub-class conditional densities
def logpdf_gau_nd(X, mu, sigma):
    fir = X.shape[0] * np.log(2 * np.pi)
    sec = np.linalg.slogdet(sigma)[1]
    x_minus_mu = X - mu
    thi = (np.dot(x_minus_mu.T, np.linalg.inv(sigma)).T * x_minus_mu).sum(axis=0)

    return - 0.5 * (fir + sec + thi)