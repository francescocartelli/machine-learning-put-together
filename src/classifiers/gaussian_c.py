import numpy as np
from scipy.special import logsumexp
from utils import colm, mu_sigma, logpdf_gau_nd

from graphs import Classifier


# Gaussian classifiers
class Gaussian(Classifier):
    def __init__(self, model="MVG"):
        if model not in ["MVG", "TCG", "NBG"]:
            raise Exception(f"Model {model} not recognised.")
        self.model = model

        self.classes = None
        self.n_classes = None       # Number of classes
        self.mu = None              # Vector of means matrices
        self.sigma = None           # Vector of covariance matrices

    # Train model
    def train(self, x, y, classes=None):
        self.classes = np.unique(y) if classes is None else classes
        self.n_classes = len(self.classes)
        self.mu = np.zeros([self.n_classes, x.shape[0], 1])
        self.sigma = np.zeros([self.n_classes, x.shape[0], x.shape[0]])

        freq_c = np.zeros([self.n_classes, 1, 1])   # Frequency of each class
        for i, c in enumerate(self.classes):
            self.mu[i, :, :], sigma_i = mu_sigma(x[:, y == c])
            self.sigma[i, :, :] = np.diag(np.diag(sigma_i)) if self.model == "NBG" else sigma_i
            freq_c[i, :, :] = x[:, y == c].shape[1] / x.shape[1]  # Used only for TCG

        if self.model == "TCG":
            self.sigma = np.zeros([self.n_classes, x.shape[0], x.shape[0]]) + (self.sigma * freq_c).sum(axis=0)

    # Given an evaluation set returns the log likelihood for all the sample and for all the classes
    def log_l(self, x):
        return np.array([logpdf_gau_nd(x, self.mu[i, :, :], self.sigma[i, :, :]) for i, c in enumerate(self.classes)])

    # Given a vector of log likelihood and a fixed prior returns the posterior probabilities
    def posterior_log_l(self, log_l, priors):
        joint_p = log_l + colm(np.log(priors))
        return joint_p - logsumexp(joint_p, axis=0)

    # Given an evaluation set returns the log likelihood and posterior probability if priors is not none
    def transform(self, x):
        return self.log_l(x)

    def __str__(self):
        return f"Gaussian({self.model})"
