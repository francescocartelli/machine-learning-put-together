import numpy as np
from scipy.special import logsumexp
from utils import colm, mu_sigma, logpdf_gau_nd

from graphs import *


# Marginal log-densities
def logpdf_gmm(X, gmm):
    return logsumexp(joint_density_gmm(X, gmm), axis=0)


# Joint log-densities, just the sub-class conditional densities + the log-prior
def joint_density_gmm(X, gmm):
    return np.array([logpdf_gau_nd(X, mu, sigma) + np.log(w) for (w, mu, sigma) in gmm])


# Combination of the two methods above
def joint_density_and_logpdf_gmm(X, gmm):
    joint_d = joint_density_gmm(X, gmm)
    return joint_d, logsumexp(joint_d, axis=0)


# Posterior probability from joint distribution and marginal distribution
def posterior_p_gmm(joint_d, margin_d):
    return np.exp(joint_d - margin_d)


def avoid_cov_deg(C, psi=0):
    U, s, _ = np.linalg.svd(C)
    s[s < psi] = psi
    C = np.dot(U, colm(s)*U.T)
    return C


def split_gmm(gmm, alpha=0.1):
    w, mu, sigma = gmm
    U, s, Vh = np.linalg.svd(sigma)
    d = U[:, 0:1] * s[0] ** 0.5 * alpha
    return (w/2, mu+d, sigma), (w/2, mu-d, sigma)


# Expectation maximization algorithm for the maximization of GMM log-likelihood
def gmm_em(X, gmm, threshold=10**-6, psi=0, model="FCG"):
    ll_prev = None

    while(True):
        # E-step
        joint_d, margin_d = joint_density_and_logpdf_gmm(X, gmm)
        post_p = posterior_p_gmm(joint_d, margin_d)

        # Conditional stop
        ll_act = np.mean(margin_d)
        if ll_prev is not None and ll_act-ll_prev < threshold:
            break
        ll_prev = ll_act

        # M-step
        Z_g = np.sum(post_p, axis=1).reshape(post_p.shape[0], 1)
        F_g = np.dot(post_p, X.T)
        S_g = np.array([np.dot(post_p[g] * X, X.T) for g in range(len(gmm))])

        mu_g = F_g / Z_g
        sigma_g = np.array([(S_g[g] / Z_g[g]) - np.dot(colm(mu_g[g]), colm(mu_g[g]).T) for g in range(len(gmm))])
        w_g = Z_g / Z_g.sum()

        if model == 'FCG' or model == 'NBG':
            for i in range(len(gmm)):
                sigma_g[i] = np.diag(np.diag(sigma_g[i])) if model == 'NBG' else sigma_g[i]
                sigma_g[i] = avoid_cov_deg(sigma_g[i], psi)
        elif model == 'TCG':
            sigma_m = np.array([Z_g[g] * sigma_g[g] for g in range(len(gmm))]).sum(axis=0) / X.shape[1]     # Sigma mean
            sigma_m = avoid_cov_deg(sigma_m, psi)
            sigma_g = [sigma_m] * len(sigma_g)

        for g in range(len(gmm)):
            gmm[g] = (w_g[g, :].item(), mu_g[g].reshape(mu_g.shape[1], 1), sigma_g[g])

    return gmm, post_p, margin_d


# LBG algorithm used for increase the number of GMM components
def gmm_lbg(X, G, threshold=10**-6, alpha=0.1, psi=0.01, model="FCG"):
    mu, sigma = mu_sigma(X)
    sigma = avoid_cov_deg(sigma, psi)
    sigma = np.diag(np.diag(sigma)) if model == "NBG" else sigma
    gmm = [(1.0, mu, sigma)]    # Initialization of GMM with 1 component

    # In case of G==1 we still have post_p and margin_d
    joint_d, margin_d = joint_density_and_logpdf_gmm(X, gmm)
    post_p = posterior_p_gmm(joint_d, margin_d)

    # At each iteration components number is duplicated
    for _ in range(int(np.log2(G))):
        gmm_act = [split_gmm(gmm[g], alpha=alpha) for g in range(len(gmm))]                     # It is a list of couples
        gmm_act = list(sum(gmm_act, ()))                                                        # From list of couples to flat list
        gmm, post_p, margin_d = gmm_em(X, gmm_act, threshold=threshold, psi=psi, model=model)   # GMM-EM maximization for the new components

    return gmm, post_p, margin_d


class GMM(Classifier):
    def __init__(self, n=4, alpha=0.1, psi=0.01, model="FCG"):
        self.gmm = None                     # GMM components each one is a triplet (w, mu, sigma)
        self.n = n
        self.alpha = alpha
        self.psi = psi
        self.classes = None
        self.n_classes = None
        self.model = model
        if self.model not in ["FCG", "TCG", "NBG"]:
            raise Exception(f"Model {self.model} not recognised.")

    # Train model
    def train(self, x, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        self.gmm = []   # Empty the state in case of previous training existence
        for c in self.classes:
            gmm_c, _, _ = gmm_lbg(x[:, y == c], self.n, alpha=self.alpha, psi=self.psi, model=self.model)
            self.gmm.append(gmm_c)

    # Given an evaluation set returns the log likelihood for all the sample and for all the classes
    def transform(self, x):
        log_l = np.array([logpdf_gmm(x, self.gmm[g]) for g in range(len(self.gmm))])
        return log_l

    # Given a vector of log likelihood and a fixed prior returns the posterior probabilities
    def posterior_log_l(self, log_l, priors):
        joint_p = log_l + np.log(priors)
        return joint_p - logsumexp(joint_p, axis=0)

    def __str__(self):
        return f"GMM(n: {self.n} alpha:{self.alpha} psi:{self.psi}"
