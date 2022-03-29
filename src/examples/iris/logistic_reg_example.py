from examples.iris.data_util import load_iris_binary_split
from src.classifiers.logistic_reg import LogisticReg


# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    LOG_REG = LogisticReg(DTR, LTR)

    for l in [0, 10**(-6), 10**(-3), 1.0]:
        LOG_REG.train_logistic_reg(l, 0.5)
        log_l = LOG_REG.log_l_ratio(DTE)

        print(f"LogReg (l:{l}) error rate:", str(((log_l > 0) != LTE).sum() / LTE.size * 100) + "%")