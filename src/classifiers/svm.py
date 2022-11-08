import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from src.utils import colm, rowm
from src.graphs import *


# Support Vector Machine classifier
class SVM(Classifier):
    def __init__(self, C=1, K=1, kernel=None, prior=None):
        self.C, self.K = C, K       # C and K parameters
        self.Z = None
        self.x_K = None
        self.kernel = kernel        # Kernel object with function f of two numeric variables
        self.alpha = None           # Alpha vector, set in training
        self.W = None               # Weight matrix, set in training, if SVM has linear kernel
        self.H = None
        self.prior = prior
        self.x_ = None

    # Train SVM classifier, prior is passed in training
    def train(self, x, y, x0=None, factr=1.0, maxfun=15000):
        self.Z = colm(y * 2 - 1)    # Labels column matrix where classes are -1 and 1

        self.x_ = x

        # Computing H matrix, if kernel is None the SVM has a linear kernel
        if self.kernel is None:
            self.x_K = np.concatenate((x, rowm(np.zeros(x.shape[1])) + self.K), axis=0)  # Train data plus K row
            G = np.dot(self.x_K.T, self.x_K)    # G is dot product between data+K and its transposed
        else:
            G = [[self.kernel.f(x.T[i], x.T[j]) for j in range(self.Z.shape[0])] for i in range(self.Z.shape[0])]
        self.H = G * self.Z * self.Z.T

        # Box constraints definition with regularization term
        if self.prior is None:
            bounds = [(0, self.C) for _ in range(x.shape[1])]
        else:
            prior_emp = (1.0*(self.Z > 0)).sum()/y.size
            CT, CF = self.C*self.prior/prior_emp, self.C*(1-self.prior)/(1-prior_emp)
            bounds = [(0, CF) if y[i] == 0 else (0, CT) for i in range(x.shape[1])]

        x0 = colm(np.zeros(x.shape[1])) if x0 is None else x0

        # Optimize alpha matrix
        self.alpha, _, _ = fmin_l_bfgs_b(self.dual_loss, x0, bounds=bounds, approx_grad=False, iprint=-1, factr=factr, maxfun=maxfun)

        # Update weight matrix if the SVM has linear kernel
        if self.kernel is None:
            self.W = np.sum(colm(self.alpha) * self.Z * self.x_K.T, axis=0)

    # Loss function for the dual problem
    def dual_loss(self, alpha):
        alpha = colm(alpha)
        ones = colm(np.ones(alpha.size))
        return 0.5 * np.dot(np.dot(alpha.T, self.H), alpha) - np.dot(alpha.T, ones), np.dot(self.H, alpha) - ones

    # Given an evaluation set returns the scores
    def transform(self, x):
        if self.kernel is None:
            DTE_k = np.vstack([x, np.zeros([1, x.shape[1]]) + self.K])
            return np.dot(self.W.T, DTE_k)
        else:
            # Matrix of values as kernel function (D[i], DTE[j])
            kernel_DTE = np.array([np.array([self.kernel.f(self.x_.T[j], x.T[i])
                                              for j in range(self.x_.shape[1])])
                                   for i in range(x.shape[1])])
            return (colm(self.alpha) * self.Z * kernel_DTE.T).sum(axis=0)

    def __str__(self):
        return f"SVM(C:{self.C}, K:{self.K}, kernel:{self.kernel.__str__()}, prior:{self.prior})"


# Encapsulation of polynomial kernel function with bias and degree terms
class Poly:
    def __init__(self, c, d):
        self.c, self.d = c, d

    def f(self, x1, x2):
        return (np.dot(x1.T, x2) + self.c) ** self.d

    def __str__(self):
        return f"Poly (c:{self.c}, d:{self.d})"


# Encapsulation of radial basis kernel function with lambda and bias terms
class RBF:
    def __init__(self, l, k):
        self.l, self.k = l, k

    def f(self, x1, x2):
        return np.exp(-self.l * np.linalg.norm(x1 - x2)**2) + self.k

    def __str__(self):
        return f"RBF (l:{self.l}, K:{self.k})"
