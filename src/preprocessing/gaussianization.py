from scipy.stats import norm
import numpy as np

from graphs import *


def gaussianization(D):
    D_g = np.array([((-D[i, :]).argsort()).argsort() + 1 for i in range(D.shape[0])])
    D_g = D_g / (D_g.shape[1] + 2)
    D_g = np.array([norm.ppf(D_g[i, :]) for i in range(D.shape[0])])

    return D_g


class Gaussianization(Node):
    def __init__(self):
        self.train = None

    def fit(self, x):
        self.train = x

    def transform(self, x):
        return gaussianization(np.concatenate((self.train, x), axis=1))[:, self.train.shape[1]:]

    def __str__(self):
        return f"Gau"




