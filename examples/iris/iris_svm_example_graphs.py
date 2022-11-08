from examples.iris.iris_data_utils import load_iris_binary_split
from src.graphs import *
from src.classifiers import *
from src.plotting import *


if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    g = Graph()
    svms = g.add_multiple(SVM, Grid(K=[1, 10], C=[0.1, 1.0, 10.0]), inputs=[g.x, g.y])
    svms_kernels = g.add_multiple(SVM, Grid(K=[0, 1.0], kernel=[Poly(0, 2), Poly(1, 2), RBF(1, 0), RBF(10, 0)]), inputs=[g.x, g.y])
    pri = g.add(StandardPrinter, printAccuracy=True, inputs=[*svms, *svms_kernels])

    g.fit(DTR, LTR)
    g.transform(DTE)
    g.output(LTE)


