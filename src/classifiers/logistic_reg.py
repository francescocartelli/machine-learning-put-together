import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from graphs import *


# Logistic Regression binary classifier
class LogisticReg(Classifier):
    def __init__(self, l=10**-3, prior=None):
        self.l = l
        self.prior = prior
        self.w = None                       # Optimized weights vector, set in training
        self.b = None                       # Optimized biases vector, set in training

    # Loss function standard
    def j_loss(self, v, x, y):
        w_, b_ = v[0:-1], v[-1]
        exps = np.array([np.dot(w_.T, x[:, i]) + b_ for i in range(x.shape[1])])
        logY = np.log(1 + np.e ** (-(y * 2 - 1) * exps))
        return (self.l * (w_ * w_).sum() / 2) + logY.mean()

    # Loss function regularized with priors
    def j_loss_reg(self, v, x, y):
        w_, b_ = v[0:-1], v[-1]
        exps = np.array([np.dot(w_.T, x[:, i]) + b_ for i in range(x.shape[1])])
        logY = np.log1p(np.exp(-(y * 2 - 1) * exps))
        logY_0_m = logY[y == 0].mean() * (1 - self.prior)
        logY_1_m = logY[y == 1].mean() * self.prior

        return (self.l * (w_ * w_).sum() / 2) + logY_0_m + logY_1_m

    # Train binary logistic regression with lambda as hyperparameter
    def train(self, x, y):
        x0 = np.zeros(x.shape[0] + 1)

        # Regularization requires prior not None
        if self.prior is None:
            x, value, _ = fmin_l_bfgs_b(self.j_loss, x0, args=(x, y), approx_grad=True, iprint=0)
        else:
            x, value, _ = fmin_l_bfgs_b(self.j_loss_reg, x0, args=(x, y), approx_grad=True, iprint=0)

        self.w, self.b = x[0:-1], x[-1]

    # Given an evaluation set returns the log likelihood ratio
    def transform(self, x):
        return np.dot(self.w.T, x) + self.b

    def __str__(self):
        return f"LogReg(l:{self.l}, prior:{self.prior})"