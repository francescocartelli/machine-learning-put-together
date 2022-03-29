from scipy.stats import norm
import numpy as np


def gaussianization(D):
    D_g = np.array([(D[i, :].argsort()).argsort() + 1 for i in range(D.shape[0])])
    D_g = D_g / (D_g.shape[1] + 2)

    return norm.ppf(D_g, loc=0, scale=1)