import numpy as np
from graphs import *
from measuring_predictions import *


def accuracy(scores, labels, round=3):
    if len(scores.shape) == 1:
        return np.round(((scores > 0) == labels).sum() / labels.size, round)
    else:
        return np.round((np.argmax(scores, axis=0) == labels).sum() / labels.size, round)


class StandardPrinter(Printer):
    def __init__(self, print=True, output_dir=None, accuracy=True, priors=None, actual=False, round=3):
        self.print = print
        self.output_dir = output_dir
        self.round = round
        self.accuracy = accuracy
        self.priors = priors
        self.actual = actual

    def __call__(self, *args, **kwargs):
        nodes = kwargs["nodes"]
        scores = np.array(kwargs["scores"], dtype=object)
        labels = np.array(kwargs["labels"])

        for i, node in enumerate(nodes):
            if self.accuracy:
                print(node, f"accuracy: {accuracy(scores[i], labels, self.round) * 100}%")

            if self.priors is not None:
                sc = scores[i][1] - scores[i][0] if len(scores[i].shape) > 1  else scores[i]

                min_dcfs = [np.round(min_dcf(sc, labels, prior, 1, 1), self.round) for prior in self.priors]
                act_dcfs = [np.round(norm_dcf_threshold(sc, labels, prior, 1, 1), self.round) for prior in self.priors]

                print(node, "minDCF", min_dcfs)
                if self.actual:
                    print(node, "actDCF", act_dcfs)
