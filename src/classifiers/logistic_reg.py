import numpy as np
from scipy.optimize import fmin_l_bfgs_b


# Logistic Regression binary classifier
class LogisticReg:
    def __init__(self, D, L, classes=None):
        self.D, self.L = D, L               # Training data and labels
        if classes is None:
            self.classes = np.unique(L)     # Classes involved
        else:
            self.classes = classes
        self.w = None                       # Optimized weights vector, set in training
        self.b = None                       # Optimized biases vector, set in training

    # Loss function standard
    def j_loss(self, v, l):
        w_, b_ = v[0:-1], v[-1]
        exps = np.array([np.dot(w_.T, self.D[:, i]) + b_ for i in range(self.D.shape[1])])
        logY = np.log(1 + np.e ** (-(self.L * 2 - 1) * exps))
        return (l * (w_ * w_).sum() / 2) + logY.mean()

    # Loss function regularized with priors
    def j_loss_reg(self, v, l, prior):
        w_, b_ = v[0:-1], v[-1]
        exps = np.array([np.dot(w_.T, self.D[:, i]) + b_ for i in range(self.D.shape[1])])
        logY = np.log1p(np.exp(-(self.L * 2 - 1) * exps))
        logY_0_m = logY[self.L == 0].mean() * (1 - prior)
        logY_1_m = logY[self.L == 1].mean() * prior

        return (l * (w_ * w_).sum() / 2) + logY_0_m + logY_1_m

    # Train binary logistic regression with lambda as hyperparameter
    def train_logistic_reg(self, l, prior=None):
        x0 = np.zeros(self.D.shape[0] + 1)

        # Regularization requires prior not None
        if prior is None:
            x, value, _ = fmin_l_bfgs_b(self.j_loss, x0, args=[l], approx_grad=True, iprint=0)
        else:
            x, value, _ = fmin_l_bfgs_b(self.j_loss_reg, x0, args=(l, prior), approx_grad=True, iprint=0)

        self.w, self.b = x[0:-1], x[-1]

    # Given an evaluation set returns the log likelihood ratio
    def log_l_ratio(self, DTE):
        return np.dot(self.w.T, DTE) + self.b