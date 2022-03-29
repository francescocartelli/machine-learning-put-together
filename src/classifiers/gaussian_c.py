import numpy as np
from scipy.special import logsumexp
from utils import colm, mu_sigma, logpdf_gau_nd


# Gaussian classifiers
class Gaussian:
    def __init__(self, D, L, classes=None):
        self.D, self.L = D, L                                                       # Training data and labels
        if classes is None:
            self.classes = np.unique(L)                                             # Classes involved
        else:
            self.classes = classes
        self.n_classes = len(self.classes)                                          # Number of classes
        self.mu = np.zeros([self.n_classes, self.D.shape[0], 1])                    # Vector of means matrices
        self.sigma = np.zeros([self.n_classes, self.D.shape[0], self.D.shape[0]])   # Vector of covariance matrices

    # Train model
    def train(self, model="MVG"):
        if model not in ["MVG", "TCG", "NBG"]:
            raise Exception(f"Model {model} not recognised.")

        freq_c = np.zeros([self.n_classes, 1, 1])   # Frequency of each class
        for i, c in enumerate(self.classes):
            self.mu[i, :, :], sigma_i = mu_sigma(self.D[:, self.L == c])
            self.sigma[i, :, :] = np.diag(np.diag(sigma_i)) if model == "NBG" else sigma_i
            freq_c[i, :, :] = self.D[:, self.L == c].shape[1] / self.D.shape[1]  # Used only for TCG

        if model == "TCG":
            self.sigma = np.zeros([self.n_classes, self.D.shape[0], self.D.shape[0]]) + (self.sigma * freq_c).sum(axis=0)

    # Given an evaluation set returns the log likelihood for all the sample and for all the classes
    def log_l(self, DTE):
        return np.array([logpdf_gau_nd(DTE, self.mu[i, :, :], self.sigma[i, :, :]) for i in self.classes])

    # Given a vector of log likelihood and a fixed prior returns the posterior probabilities
    def posterior_log_l(self, log_l, priors):
        joint_p = log_l + colm(np.log(priors))
        return joint_p - logsumexp(joint_p, axis=0)

    # Given an evaluation set returns the log likelihood and posterior probability if priors is not none
    def evaluate(self, DTE, priors):
        log_l = self.log_l(DTE)
        return log_l, self.posterior_log_l(log_l, priors)