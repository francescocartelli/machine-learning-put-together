from examples.iris.data_utils import load_iris_binary_split
from src.classifiers.logistic_reg import LogisticReg
from preprocessing import *
from plotting import *
from measuring_predictions import *

# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    for l in [0, 10**(-6), 10**(-3), 1.0]:
        LOG_REG = LogisticReg(l)
        LOG_REG.train(DTR, LTR)
        log_l = LOG_REG.transform(DTE)

        print(f"LogReg (l:{l}) error rate:", str(((log_l > 0) != LTE).sum() / LTE.size * 100) + "%")
    print("==============================")

    pca_2 = PCA(2)
    DTR_PCA_2 = pca_2.fit_transform(DTR)
    DTE_PCA_2 = pca_2.transform(DTE)

    for l in [0, 10**(-6), 10**(-3), 1.0]:
        LOG_REG = LogisticReg(l=l)
        LOG_REG.train(DTR_PCA_2, LTR)
        log_l = LOG_REG.transform(DTE_PCA_2)

        print(f"LogReg (l:{l}, PCA: 2) error rate:", str(((log_l > 0) != LTE).sum() / LTE.size * 100) + "%")
    print("==============================")

    lda_2 = LDA(2)
    DTR_LDA_2 = lda_2.fit_transform(DTR, LTR)
    DTE_LDA_2 = lda_2.transform(DTE)

    for l in [0, 10**(-6), 10**(-3), 1.0]:
        LOG_REG = LogisticReg(l=l)
        LOG_REG.train(DTR_LDA_2, LTR)
        log_l = LOG_REG.transform(DTE_LDA_2)

        print(f"LogReg (l:{l}, LDA: 2) error rate:", str(((log_l > 0) != LTE).sum() / LTE.size * 100) + "%")
