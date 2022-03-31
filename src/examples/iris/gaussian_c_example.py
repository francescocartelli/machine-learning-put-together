import numpy as np
from examples.iris.data_utils import load_iris, load_iris_split
from src.classifiers.gaussian_c import Gaussian
from utils import KFoldCrossVal

# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_split()

    # Standard split
    G = Gaussian(DTR, LTR)
    priors = np.array([1/3, 1/3, 1/3])
    for model in ["MVG", "TCG", "NBG"]:
        G.train(model=model)
        _, posterior_prob = G.evaluate(DTE, priors)

        print(f"{model} accuracy: {(np.argmax(posterior_prob, 0) != LTE).sum() / LTE.size * 100}%")
    print("==============================")

    # Leave one out split
    D, L = load_iris()
    n_folds = 150
    k_fold = KFoldCrossVal(D, L, n_folds)
    for model in ["MVG", "TCG", "NBG"]:
        errors, eval_samples = 0, 0
        for i in range(n_folds):
            (DTR, LTR), (DTE, LTE) = k_fold.train_eval(i)
            G = Gaussian(DTR, LTR)
            G.train(model=model)
            _, posterior_prob = G.evaluate(DTE, priors)
            errors += (np.argmax(posterior_prob, 0) != LTE).sum()
            eval_samples += LTE.size
        print(f"{model} leave-one-out error: {errors / eval_samples * 100}%")