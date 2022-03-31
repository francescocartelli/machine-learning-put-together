from examples.iris.data_utils import load_iris_binary_split
from src.classifiers.svm import SVM, Poly, RBF

# Examples of training and test
if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    for K in [1, 10]:
        for C in [0.1, 1.0, 10.0]:
            S = SVM(DTR, LTR, K=K, C=C)
            S.train()
            score = S.evaluate(DTE)
            print(f"SVM (K:{K}, C:{C}) error rate:", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")

    print("\nPoly Kernels:")
    for K in [0, 1.0]:
        poly = Poly(0, 2)
        S = SVM(DTR, LTR, K=K, C=1, kernel=poly.f)
        for prior in [0.1, 0.5, 0.9]:
            S.train(prior=prior)
            score = S.evaluate(DTE)
            print(f"SVM (K:{K}, C:1, P:{prior}) error rate Poly(0, 2):", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")
    for K in [0, 1.0]:
        poly = Poly(1, 2)
        S = SVM(DTR, LTR, K=K, C=1, kernel=poly.f)
        S.train()
        score = S.evaluate(DTE)
        print(f"SVM (K:{K}, C:1) error rate Poly(1, 2):", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")

    print("\nRBF Kernels:")
    for K in [0, 1.0]:
        rbf = RBF(1, K**2)
        S = SVM(DTR, LTR, K=K, C=1, kernel=rbf.f)
        S.train()
        score = S.evaluate(DTE)
        print(f"SVM (K:{K}, C:1) error rate RBF(1):", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")
    for K in [0, 1.0]:
        rbf = RBF(10, K**2)
        S = SVM(DTR, LTR, K=K, C=1, kernel=rbf.f)
        S.train()
        score = S.evaluate(DTE)
        print(f"SVM (K:{K}, C:1) error rate RBF(10):", str(((score > 0) != LTE).sum() / LTE.size * 100) + "%")