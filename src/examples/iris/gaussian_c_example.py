import numpy as np
from examples.iris.data_utils import load_iris, load_iris_split
from src.classifiers.gaussian_c import Gaussian
from utils import KFoldCrossVal

from preprocessing import *

from scipy.stats import norm


# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_split()

    D_G_1 = gaussianization(DTR)

    # Standard split
    priors = np.array([1/3, 1/3, 1/3])
    for model in ["MVG", "TCG", "NBG"]:
        G = Gaussian(model=model)

        G.train(DTR, LTR)
        llr = G.transform(DTE)
        post_p = G.posterior_log_l(llr, priors)

        print(f"{model} accuracy: {(np.argmax(post_p, 0) != LTE).sum() / LTE.size * 100}%")
    print("==============================")

    # Leave one out split
    D, L = load_iris()
    n_folds = 150
    k_fold = KFoldCrossVal(D, L, n_folds)
    for model in ["MVG", "TCG", "NBG"]:
        errors, eval_samples = 0, 0
        for i in range(n_folds):
            (DTR, LTR), (DTE, LTE) = k_fold.train_eval(i)
            G = Gaussian(model=model)
            G.train(DTR, LTR)
            llr = G.transform(DTE)
            post_p = G.posterior_log_l(llr, priors)
            errors += (np.argmax(post_p, 0) != LTE).sum()
            eval_samples += LTE.size
        print(f"{model} leave-one-out error: {errors / eval_samples * 100}%")