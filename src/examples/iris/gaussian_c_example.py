import numpy as np
from examples.iris.data_util import load_iris_split
from src.classifiers.gaussian_c import Gaussian

# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_split()

    G = Gaussian(DTR, LTR)

    priors = np.array([1/3, 1/3, 1/3])
    for model in ["MVG", "TCG", "NBG"]:

        G.train(model=model)
        _, posterior_prob = G.evaluate(DTE, priors)
        print(f"{model} precision: {(np.argmax(posterior_prob, 0) == LTE).sum() / LTE.size * 100}%")