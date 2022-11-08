from src.graphs import *
from src.classifiers import *
from src.preprocessing import *
from src.plotting import *
from src.transformations import *
from examples.wines.wines_data_utils import get_wine_data

if __name__ == '__main__':
    DTR, LTR, DTE, LTE = get_wine_data(labels=True)
    DTR = DTR.T
    DTE = DTE.T

    g = Graph()
    std = g.add(Standardization, inputs=[g.x])
    gau = g.add(Gaussianization, inputs=[g.x])
    poly = g.add(SVM, C=1, K=1, kernel=Poly(2, 1), label="Poly", inputs=[std, g.y])
    rbf = g.add(SVM, C=0.1, kernel=RBF(0.5, 0.1), label="RBF", inputs=[std, g.y])
    mvg = g.add(Gaussian, label="MVG", inputs=[gau, g.y])
    gmm = g.add(GMM, n=8, label="GMM", inputs=[std, g.y])
    sta = g.add(Stack, inputs=[poly, rbf, mvg, gmm])
    rec = g.add(LogisticReg, recal=True, label="LR", inputs=[sta, g.y])

    g.add(StandardPrinter, printAccuracy=True, priors=[0.5, 0.1, 0.9],
          printMinDCF=True, printActDCF=True, output_dir='./results/fusion',
          inputs=[rec])

    g.display()

    g.fit(DTR, LTR)
    g.transform(DTE)
    g.output(LTE)
