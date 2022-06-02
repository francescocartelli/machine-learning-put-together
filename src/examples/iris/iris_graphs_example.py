import numpy as np

from graphs.graphs import *
from classifiers import *
from preprocessing import *
from plotting import *
from examples.iris.iris_data_utils import load_iris_binary_split


class Stack(Transformation):
    def __init__(self, direction='v'):
        if direction != 'v' and direction != 'h':
            raise Exception(f"Stack vertically with 'v' or horizontally with 'h', receive {direction} as parameter")
        self.direction = direction

    def transform(self, *x):
        x = [x_[1] - x_[0] if len(x_.shape) > 1 else x_ for x_ in x]
        return np.vstack(x) if self.direction == 'v' else np.hstack

    def __str__(self):
        return "+"


if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    g = Graph()

    std = g.add(Standardization, inputs=[g.x])
    gau = g.add(Gaussianization, inputs=[g.x])
    poly = g.add(SVM, C=0.1, kernel=Poly(2, 1), label="Poly", inputs=[std, g.y])
    rbf = g.add(SVM, C=0.1, kernel=RBF(0.5, 0.1), label="RBF", inputs=[std, g.y])
    gmm = g.add(GMM, inputs=[std, g.y])
    mvg = g.add(Gaussian, model="MVG", inputs=[gau, g.y])
    g.add(StandardPrinter, printAccuracy=True, priors=[0.5, 0.1, 0.9], printMinDCF=True, printActDCF=True, inputs=[poly, rbf, gmm, mvg])
    g.add(BayesErrorPlotter, logPriors=np.linspace(-4, 4, 51), inputs=[poly, rbf])

    # Training
    g.fit(DTR, LTR)

    # Evaluation
    g.transform(DTE)
    g.output(LTE)

