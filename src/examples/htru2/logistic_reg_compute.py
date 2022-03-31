from classifiers import *
from measuring_predictions import *
from plotting import *
from examples.htru2.data_utils import *
from preprocessing import *

output_dir = "./outputs/"


def get_mindcf_logreg(ls, DTR, LTR, DTE, LTE):
    minDCFs = []

    for l in ls:
        lam = 10.0**l
        logisticReg = LogisticReg(DTR, LTR)
        logisticReg.train(l=lam)
        log_l_ratio = logisticReg.log_l_ratio(DTE)
        minDCFs.append(min_dcf(log_l_ratio, LTE, 0.5, 1, 1))
        print(f"l: {l} executed now")
    return minDCFs


if __name__ == "__main__":
    D, L = split_d_l('data/HTRU_2.csv', ratio=0.2)
    (DTR, LTR), (DTE, LTE) = split_tr_te(D, L, 0.9)

    ls = np.array([-i for i in range(0, 7)])
    mindcf = get_mindcf_logreg(ls, DTR, LTR, DTE, LTE)

    pca_6 = PCA(6)
    DTR_PCA_6 = pca_6.fit_transform(DTR)
    mindcf_pca_6 = get_mindcf_logreg(ls, DTR_PCA_6, LTR, pca_6.transform(DTE), LTE)

    plot_multiple_mindcf_bar_chart([mindcf, mindcf_pca_6], ls, legend=["LogReg", "LogReg PCA-6"], x_label="exp of lambda")
