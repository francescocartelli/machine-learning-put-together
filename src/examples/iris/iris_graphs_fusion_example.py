import numpy as np

from graphs.graphs import *
from classifiers import *
from preprocessing import *
from plotting import *
from transformations import *
from examples.iris.iris_data_utils import load_iris_binary_split

if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    g = Graph()

    std = g.add(Standardization, inputs=[g.x])
    poly = g.add(SVM, C=0.1, kernel=Poly(2, 1), label="Poly", inputs=[std, g.y])
    rbf = g.add(SVM, C=0.1, kernel=RBF(0.5, 0.1), label="RBF", inputs=[std, g.y])
    gmm = g.add(GMM, label="GMM", inputs=[std, g.y])
    mvg = g.add(Gaussian, model="MVG", label="MVG", inputs=[g.x, g.y])
    sta = g.add(Stack, inputs=[poly, rbf, gmm, mvg])
    rec = g.add(LogisticReg, recal=True, label="Rec", inputs=[sta, g.y])
    g.add(StandardPrinter, printAccuracy=True, priors=[0.5, 0.1, 0.9], printMinDCF=True, printActDCF=True, inputs=[poly, rbf, gmm, mvg, rec])
    g.add(BayesErrorPlotter, logPriors=np.linspace(-4, 4, 51), inputs=[rec])

    # Training
    g.fit(DTR, LTR)

    # Evaluation
    g.transform(DTE)
    g.output(LTE)

