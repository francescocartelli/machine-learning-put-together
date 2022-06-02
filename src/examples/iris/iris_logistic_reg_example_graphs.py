from examples.iris.iris_data_utils import load_iris_binary_split
from graphs import *
from classifiers import *
from plotting import *


if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    g = Graph()
    lregs = g.add_multiple(LogisticReg, Grid(l=[0, 10**-6, 10**-3, 1.0]), inputs=[g.x, g.y])
    pri = g.add(StandardPrinter, printAccuracy=True, inputs=[*lregs])

    g.fit(DTR, LTR)
    g.transform(DTE)
    g.output(LTE)


