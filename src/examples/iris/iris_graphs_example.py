import numpy as np

from graphs.graphs import *
from classifiers import *
from preprocessing import *
from plotting import *
from examples.iris.iris_data_utils import load_iris_binary_split


if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    g = Graph()
    std = g.add(Standardization, inputs=[g.x])
    gau = g.add(Gaussianization, inputs=[g.x])
    svm = g.add(SVM, C=0.1, inputs=[std, g.y])
    log = g.add(LogisticReg, inputs=[std, g.y])
    mvg = g.add(Gaussian, model="MVG", inputs=[gau, g.y])
    gmm = g.add(GMM, inputs=[gau, g.y])
    prt = g.add(StandardPrinter, accuracy=True, priors=[0.5, 0.1, 0.9], actual=True, inputs=[mvg, log, svm, gmm])

    g.display()

    g.fit(DTR, LTR)
    g.transform(DTE)
    g.output(LTE)

