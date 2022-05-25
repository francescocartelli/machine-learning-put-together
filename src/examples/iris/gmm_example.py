from examples.iris.data_utils import load_iris_split
from src.classifiers.gmm import GMM

import numpy as np

# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_split()
    for model in ["FCG", "NBG", "TCG"]:
        for n_components in [1, 2, 4, 8, 16]:
            gmm = GMM(n=n_components, alpha=0.1, psi=0.01, model=model)
            gmm.train(DTR, LTR)
            predicted_labels = gmm.transform(DTE).argmax(axis=0)
            print(f"MMG ({n_components} components, {model} model) error: {(predicted_labels != LTE).sum() / LTE.size * 100}%")
        print("==============================")
