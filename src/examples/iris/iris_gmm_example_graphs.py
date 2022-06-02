from examples.iris.iris_data_utils import load_iris_split
from graphs import *
from classifiers import *
from plotting import *


if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_split()

    g = Graph()
    gmms = g.add_multiple(GMM, Grid(model=["FCG", "NBG", "TCG"], n=[1, 2, 4, 8, 16]), inputs=[g.x, g.y])
    pri = g.add(StandardPrinter, printAccuracy=True, inputs=[*gmms])

    g.fit(DTR, LTR)
    g.transform(DTE)
    g.output(LTE)

