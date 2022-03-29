import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from utils import colm, rowm


# Support Vector Machine classifier
class SVM:
    def __init__(self, D, L, C=1, K=1, prior=None, kernel=None):
        self.D, self.L = D, L       # Training data and labels
        self.C, self.K = C, K       # C and K parameters
        self.Z = colm(L * 2 - 1)    # Labels column matrix where classes are -1 and 1
        self.kernel = kernel        # Kernel function of two numeric variable
        self.alpha = None           # Alpha vector, set in training
        self.W = None               # Weight matrix, set in training, if SVM has linear kernel

        # Computing H matrix, if kernel is None the SVM has a linear kernel
        if kernel is None:
            self.D_K = np.concatenate((D, rowm(np.zeros(D.shape[1])) + K), axis=0)  # Train data plus K row
            G = np.dot(self.D_K.T, self.D_K)    # G is dot product between data+K and its transposed
        else:
            G = [[self.kernel(D.T[i], D.T[j]) for j in range(self.Z.shape[0])] for i in range(self.Z.shape[0])]
        self.H = G * self.Z * self.Z.T

    # Train SVM classifier, prior is passed in training
    def train_svm(self, x0=None, prior=None, factr=1.0, maxfun=15000):
        # Box constraints definition with regularization term
        if prior is None:
            bounds = [(0, self.C) for _ in range(self.D.shape[1])]
        else:
            prior_emp = (1.0*(self.Z > 0)).sum()/self.L.size
            CT, CF = self.C*prior/prior_emp, self.C*(1-prior)/(1-prior_emp)
            bounds = [(0, CF) if self.L[i] == 0 else (0, CT) for i in range(self.D.shape[1])]

        x0 = colm(np.zeros(self.D.shape[1])) if x0 is None else x0

        # Optimize alpha matrix
        self.alpha, _, _ = fmin_l_bfgs_b(self.dual_loss, x0, bounds=bounds, approx_grad=False, iprint=0, factr=factr, maxfun=maxfun)

        # Update weight matrix if the SVM has linear kernel
        if self.kernel is None:
            self.W = np.sum(colm(self.alpha) * self.Z * self.D_K.T, axis=0)

    # Loss function for the dual problem
    def dual_loss(self, alpha):
        alpha = colm(alpha)
        ones = colm(np.ones(alpha.size))
        return 0.5 * np.dot(np.dot(alpha.T, self.H), alpha) - np.dot(alpha.T, ones), np.dot(self.H,alpha) - ones

    # Given an evaluation set returns the scores
    def evaluate(self, DTE):
        if self.kernel is None:
            DTE_k = np.vstack([DTE, np.zeros([1, DTE.shape[1]]) + self.K])
            return np.dot(self.W.T, DTE_k)
        else:
            # Matrix of values as kernel function (D[i], DTE[j])
            kernel_DTE = np.array([np.array([ self.kernel(self.D.T[j], DTE.T[i])
                                              for j in range(self.D.shape[1])])
                                   for i in range(DTE.shape[1])])
            return (colm(self.alpha) * self.Z * kernel_DTE.T).sum(axis=0)


# Encapsulation of polynomial kernel function with bias and degree terms
class Poly:
    def __init__(self, c, d):
        self.c, self.d = c, d

    def f(self, x1, x2):
        return (np.dot(x1.T, x2) + self.c) ** self.d


# Encapsulation of radial basis kernel function with lambda and bias terms
class RBF:
    def __init__(self, l, k):
        self.l, self.k = l, k

    def f(self, x1, x2):
        return np.exp(-self.l * np.linalg.norm(x1 - x2)**2) + self.k