import numpy as np

from graphs.graph import *
from classifiers import *
from preprocessing import *
from plotting import *
from examples.iris.data_utils import load_iris_binary_split


class C(Classifier):
    def __init__(self):
        self.s = None

    def train(self, x, y):
        self.s = x + y

    def transform(self, x):
        return self.s + x

    def train_transform(self, x, y):
        self.train(x, y)
        return self.transform(x)

    def __str__(self):
        return f"C"


class F(Transformation):
    def transform(self, x):
        return -x

    def __str__(self):
        return f"F"


class P(Transformation):
    def transform(self, x):
        print("->", x)
        return x


if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    g = Graph()
    gau = g.add(Gaussian, model="MVG", inputs=[g.get_x, g.get_y])
    std = g.add(Standardization, inputs=g.get_x)
    log = g.add(LogisticReg, inputs=[std.o, g.get_y])
    svm = g.add(SVM, C=0.1, kernel=RBF(1.0, 1.0), inputs=[std.o, g.get_y])
    prt = g.add(StandardPrinter, accuracy=False, priors=[0.5, 0.1, 0.9], actual=True, inputs=[gau, log, svm])

    g.fit(DTR, LTR)
    g.transform(DTE)
    g.output(LTE)

