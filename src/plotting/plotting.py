import matplotlib.pyplot as plt

from measuring_predictions import *
from graphs import Printer


def plot_multiple_bayes_error(S_list, labels, logPriors, legend=None):
    for i, S in enumerate(S_list):
        legend_i = legend[i] if legend is not None and i < len(legend) else None

        dcfs, min_dcfs = bayes_errors_from_priors(S, labels, logPriors)
        last_plot = plt.plot(logPriors, dcfs, label=f"DCF {legend_i}")
        # Print the min dcf with a line of the same color as the dcf but with a dashed style
        plt.plot(logPriors, min_dcfs, label=f"minDCF {legend_i}", color=last_plot[0].get_color(), linestyle="dashed")
    if legend is not None:
        plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([min(logPriors), max(logPriors)])
    plt.show()


def plot_multiple_mindcf_bar_chart(minDCFs, x, legend=None, x_label=None):
    ax = plt.subplot()
    width = 1 / (len(minDCFs) + 1)  # One bar as padding
    for i, minDCF in enumerate(minDCFs):
        legend_i = legend[i] if legend is not None and i < len(legend) else None    # Name of the model

        ax.bar(x+(width*i), minDCF, width=width, label=f"{legend_i}")  # minDCF is a vector, minDCFs is a list of vectors
    if x_label is not None:
        plt.xlabel(x_label)    # legend_x is the x-axis label
    if legend is not None:
        plt.legend()
    plt.ylabel("minDCF")
    plt.show()


def plot_roc_curve(S, labels, size=1000):
    roc_matrix = roc_curve_vector(S, labels, size)
    plt.plot(roc_matrix[:, 0], roc_matrix[:, 1])
    plt.show()


class BayesErrorPlotter(Printer):
    def __init__(self, logPriors):
        self.logPriors = logPriors

    def __call__(self, *args, **kwargs):
        nodes = kwargs.pop("nodes")
        scores = kwargs.pop("scores")
        labels = kwargs.pop("labels")

        plot_multiple_bayes_error(scores, labels, self.logPriors, legend=nodes)

