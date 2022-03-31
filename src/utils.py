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


class KFoldCrossVal:
    def __init__(self, D, L, k):
        p = np.random.permutation(L.size)
        self.D, self.L = D[:, p], L[p]          # Shuffled data and label
        self.k = k                              # Number of folds
        self.fold_size = int(self.L.size / k)   # Size of each fold

    # Get train and evalutation sets when fold n is the evaluation one (starting from 0)
    def train_eval(self, n):
        leave_out_ids = np.arange(n * self.fold_size, (n + 1) * self.fold_size, 1)  # Generate the indexes
        DTR = np.delete(self.D, leave_out_ids, axis=1)                              # Take everything except fold n
        LTR = np.delete(self.L, leave_out_ids)
        DTE = self.D[:, leave_out_ids]                                              # Take only the fold n
        LTE = self.L[leave_out_ids]
        return (DTR, LTR), (DTE, LTE)
