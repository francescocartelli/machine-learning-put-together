import numpy as np
import csv
import os

from graphs import *
from measuring_predictions import *


def accuracy(scores, labels, decimal=None):
    if len(scores.shape) == 1:
        acc = ((scores > 0) == labels).sum() / labels.size
    else:
        acc = (np.argmax(scores, axis=0) == labels).sum() / labels.size
    return np.round(acc, decimal) if decimal is not None else acc


def error_rate(scores, labels, decimal=None):
    acc = accuracy(scores, labels)
    return np.round(1 - acc, decimal) if decimal is not None else 1 - acc


class StandardPrinter(Printer):
    def __init__(self, output_dir=None, printAccuracy=True, printErrorRate=False,
                 priors=None, printMinDCF=False, printActDCF=False,
                 decimal=3):
        self.output_dir = output_dir
        self.printAccuracy = printAccuracy
        self.printErrorRate = printErrorRate

        self.priors = priors
        self.printMinDCF = printMinDCF
        self.printActDCF = printActDCF

        self.decimal = decimal

    def __call__(self, *args, **kwargs):
        nodes = kwargs["nodes"]
        scores = np.array(kwargs["scores"], dtype=object)
        labels = np.array(kwargs["labels"])

        fileDCF = None
        writer = None
        if self.output_dir is not None:
            os.mkdir(self.output_dir)
            np.save(f"{self.output_dir}/nodes.npy", np.array(nodes))
            np.save(f"{self.output_dir}/scores.npy", scores)
            np.save(f"{self.output_dir}/labels.npy", labels)
            if self.printMinDCF is not None or self.printActDCF is not None:
                fileDCF = open(f"{self.output_dir}/nodes_dcf.csv", "w")
                writer = csv.writer(fileDCF)
                writer.writerow(["Model", "metric", *self.priors])

        for i, node in enumerate(nodes):
            if self.printAccuracy:
                print(node, f"accuracy: {np.round(accuracy(scores[i], labels) * 100, self.decimal)}%")
            if self.printErrorRate:
                print(node, f"error rate: {np.round(error_rate(scores[i], labels) * 100, self.decimal)}%")

            if self.priors is not None:
                sc = scores[i][1] - scores[i][0] if len(scores[i].shape) > 1 else scores[i]

                if self.printMinDCF:
                    min_dcfs = [np.round(min_dcf(sc, labels, prior, 1, 1), self.decimal) for prior in self.priors]
                    print(node, "minDCF:", min_dcfs)

                    if writer is not None:
                        writer.writerow([node, "minDCF", *min_dcfs])
                if self.printActDCF:
                    act_dcfs = [np.round(norm_dcf_threshold(sc, labels, prior, 1, 1), self.decimal) for prior in self.priors]
                    print(node, "actDCF:", act_dcfs)

                    if writer is not None:
                        writer.writerow([node, "actDCF", *act_dcfs])

        if fileDCF is not None:
            fileDCF.close()

    def __str__(self):
        return f"Output"

