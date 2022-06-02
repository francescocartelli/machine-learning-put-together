from examples.iris.iris_data_utils import load_iris_binary_split
from src.classifiers import *

# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    for K in [1, 10]:
        for C in [0.1, 1.0, 10.0]:
            svm = SVM(K=K, C=C)
            svm.train(DTR, LTR)
            score = svm.transform(DTE)
            print(svm, "error rate:", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")
        print("==============================")

    print("\nPoly Kernels:")
    for K in [0, 1.0]:
        for prior in [0.1, 0.5, 0.9]:
            svm = SVM(K=K, C=1, kernel=Poly(0, 2), prior=prior)
            svm.train(DTR, LTR)
            score = svm.transform(DTE)
            print(svm, "error rate:", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")
    for K in [0, 1.0]:
        svm = SVM(K=K, C=1, kernel=Poly(1, 2))
        svm.train(DTR, LTR)
        score = svm.transform(DTE)
        print(svm, "error rate:", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")

    print("\nRBF Kernels:")
    for K in [0, 1.0]:
        svm = SVM(K=K, C=1, kernel=RBF(1, K ** 2))
        svm.train(DTR, LTR)
        score = svm.transform(DTE)
        print(svm, "error rate:", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")
    for K in [0, 1.0]:
        svm = SVM(K=K, C=1, kernel=RBF(10, K ** 2))
        svm.train(DTR, LTR)
        score = svm.transform(DTE)
        print(svm, "error rate:", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")