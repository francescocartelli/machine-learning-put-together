import numpy as np
import matplotlib.pyplot as plt

from examples.iris.data_util import load_iris
from src.preprocessing import *

if __name__ == "__main__":
    D, L = load_iris()

    classes = np.unique(L)

    fig, (axis0, axis1) = plt.subplots(nrows=1, ncols=2)

    # PCA example 2 components
    D_PCA = pca(D, 2)

    axis0.set_title("PCA-2")
    for i, c in enumerate(classes):
        D_PCA_c = D_PCA[:, L == c]
        axis0.scatter(D_PCA_c[0], D_PCA_c[1])

    # LDA example 2 components
    D_LDA = lda(D, L, 2)

    axis1.set_title("LDA-2")
    for i, c in enumerate(classes):
        D_LDA_c = D_LDA[:, L == c]
        axis1.scatter(D_LDA_c[0], D_LDA_c[1])

    plt.show()