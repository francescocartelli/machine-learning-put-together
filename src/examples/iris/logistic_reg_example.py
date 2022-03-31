from examples.iris.data_utils import load_iris_binary_split
from src.classifiers.logistic_reg import LogisticReg
from preprocessing import *
from plotting import *

# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    LOG_REG = LogisticReg(DTR, LTR)

    for l in [0, 10**(-6), 10**(-3), 1.0]:
        LOG_REG.train(l=l)
        log_l = LOG_REG.log_l_ratio(DTE)

        print(f"LogReg (l:{l}) error rate:", str(((log_l > 0) != LTE).sum() / LTE.size * 100) + "%")
    print("==============================")

    pca_2 = PCA(2)
    DTR_PCA_2 = pca_2.fit_transform(DTR)
    DTE_PCA_2 = pca_2.transform(DTE)

    LOG_REG = LogisticReg(DTR_PCA_2, LTR)
    for l in [0, 10**(-6), 10**(-3), 1.0]:
        LOG_REG.train(l=l)
        log_l = LOG_REG.log_l_ratio(DTE_PCA_2)

        print(f"LogReg (l:{l}, PCA: 2) error rate:", str(((log_l > 0) != LTE).sum() / LTE.size * 100) + "%")
    print("==============================")

    lda_2 = LDA(2)
    DTR_LDA_2 = lda_2.fit_transform(DTR, LTR)
    DTE_LDA_2 = lda_2.transform(DTE)

    LOG_REG = LogisticReg(DTR_LDA_2, LTR)
    for l in [0, 10**(-6), 10**(-3), 1.0]:
        LOG_REG.train(l=l)
        log_l = LOG_REG.log_l_ratio(DTE_LDA_2)

        print(f"LogReg (l:{l}, LDA: 2) error rate:", str(((log_l > 0) != LTE).sum() / LTE.size * 100) + "%")
