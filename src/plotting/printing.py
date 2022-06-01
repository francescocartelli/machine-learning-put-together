import numpy as np
from graphs import *
from measuring_predictions import *


def accuracy(scores, labels, precision=3):
    if len(scores.shape) == 1:
        return np.round(((scores > 0) == labels).sum() / labels.size, precision)
    else:
        return np.round((np.argmax(scores, axis=0) == labels).sum() / labels.size, precision)


class StandardPrinter(Printer):
    def __init__(self, output_dir=None, printAccuracy=True,
                 priors=None, printMinDCF=False, printActDCF=False,
                 precision=3):
        self.output_dir = output_dir
        self.precision = precision
        self.printAccuracy = printAccuracy

        self.priors = priors
        self.printMinDCF = printMinDCF
        self.printActDCF = printActDCF

    def __call__(self, *args, **kwargs):
        nodes = kwargs["nodes"]
        scores = np.array(kwargs["scores"], dtype=object)
        labels = np.array(kwargs["labels"])

        for i, node in enumerate(nodes):
            if self.printAccuracy:
                print(node, f"accuracy: {accuracy(scores[i], labels, self.precision) * 100}%")

            if self.priors is not None:
                sc = scores[i][1] - scores[i][0] if len(scores[i].shape) > 1  else scores[i]

                if self.printMinDCF:
                    min_dcfs = [np.round(min_dcf(sc, labels, prior, 1, 1), self.precision) for prior in self.priors]
                    print(node, "minDCF:", min_dcfs)
                if self.printActDCF:
                    act_dcfs = [np.round(norm_dcf_threshold(sc, labels, prior, 1, 1), self.precision) for prior in self.priors]
                    print(node, "actDCF:", act_dcfs)

    def __str__(self):
        return f"Output"

