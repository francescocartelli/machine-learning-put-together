from examples.iris.iris_data_utils import load_iris_binary_split
from src.classifiers import *

# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    for K in [1, 10]:
        for C in [0.1, 1.0, 10.0]:
            S = SVM(K=K, C=C)
            S.train(DTR, LTR)
            score = S.transform(DTE)
            print(SVM, "error rate:", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")
            print(SVM)
        print("==============================")

    print("\nPoly Kernels:")
    for K in [0, 1.0]:
        poly = Poly(0, 2)
        for prior in [0.1, 0.5, 0.9]:
            S = SVM(K=K, C=1, kernel=poly, prior=prior)
            S.train(DTR, LTR)
            score = S.transform(DTE)
            print(SVM, "error rate Poly(0, 2):", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")
    for K in [0, 1.0]:
        poly = Poly(1, 2)
        S = SVM(K=K, C=1, kernel=poly)
        S.train(DTR, LTR)
        score = S.transform(DTE)
        print(SVM, "error rate Poly(1, 2):", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")

    print("\nRBF Kernels:")
    for K in [0, 1.0]:
        rbf = RBF(1, K**2)
        S = SVM(K=K, C=1, kernel=rbf)
        S.train(DTR, LTR)
        score = S.transform(DTE)
        print(SVM, "error rate RBF(1):", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")
    for K in [0, 1.0]:
        rbf = RBF(10, K**2)
        S = SVM(K=K, C=1, kernel=rbf)
        S.train(DTR, LTR)
        score = S.transform(DTE)
        print(SVM, "(K:{K}, C:1) error rate RBF(10):", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")