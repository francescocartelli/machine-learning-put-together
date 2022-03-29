import numpy as np
import matplotlib.pyplot as plt

from measuring_predictions import *


def plot_multiple_dcf_mindcf(S_list, labels, logPriors, legend=None):
    for i, S in enumerate(S_list):
        dcfs, min_dcfs = bayes_errors_from_priors(S, labels, logPriors)
        last_plot = plt.plot(logPriors, dcfs, label=f"DCF {legend[i]}")
        # Print the min dcf with a line of the same color as the dcf but with a dashed style
        plt.plot(logPriors, min_dcfs, label=f"minDCF {legend[i]}", color=last_plot[0].get_color(), linestyle="dashed")
    if legend is not None:
        plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([min(logPriors), max(logPriors)])
    plt.show()
