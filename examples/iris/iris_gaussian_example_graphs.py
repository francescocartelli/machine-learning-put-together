from .iris_data_utils import load_iris_split
from src.graphs import *
from src.classifiers import *
from src.plotting import *


if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_split()

    g = Graph()
    gaus = g.add_multiple(Gaussian, Grid(model=["MVG", "NBG", "TCG"]), inputs=[g.x, g.y])
    pri = g.add(StandardPrinter, printAccuracy=True, inputs=[*gaus])

    g.fit(DTR, LTR)
    g.transform(DTE)
    g.output(LTE)


