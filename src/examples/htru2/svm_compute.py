from classifiers import *
from measuring_predictions import *
from examples.htru2.data_utils import *

output_dir = "./outputs/"

# Each llr list from the a model is saved
# Each posterior probability list from a model with specified prior is saved
# A csv file is saved:
#           |                min-dcf            |
#   model   |   p: 0.1   |   p: 0.5 |   p: 0.9  |
if __name__ == "__main__":
    D, L = split_d_l('data/HTRU_2.csv', ratio=0.2)
    (DTR, LTR), (DTE, LTE) = split_tr_te(D, L, 0.9)

    priors = [0.1, 0.5, 0.9]

    np.save(f"{output_dir}labels", LTE)         # Saving labels for protting of Bayes error graph

    mindcf_csv = open(f"{output_dir}mindcf.csv", "w")
    writer = csv.writer(mindcf_csv)
    writer.writerow(["model"] + priors)         # Write header

    for model in models:
        svm = SVM(DTR, LTR)
        g.train(model=model)
        log_l = g.log_l(DTE)
        log_l_ratio = log_l[1] - log_l[0]
        post_pr = g.posterior_log_l(log_l, 0.5)
        np.save(f"{output_dir}{model}_llr", post_pr[1] - post_pr[0])
        row = [model]
        for prior in priors:
            post_p = g.posterior_log_l(log_l, [prior, 1 - prior])
            score = post_p[1] - post_p[0]
            row.append(min_dcf(score, LTE, prior, 1, 1))
            #np.save(f"{output_dir}{model}_{int(prior * 10)}_score", score)
        writer.writerow(row)

    mindcf_csv.close()
