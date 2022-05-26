import numpy as np
from graphs import *


def accuracy(scores, labels, round=3):
    if len(scores.shape) == 1:
        return np.round(((scores > 0) == labels).sum() / labels.size, round)
    else:
        return np.round((np.argmax(scores, axis=0) == labels).sum() / labels.size, round)


class StandardPrinter(Printer):
    def __init__(self, print=True, output_dir=None, round=3):
        self.print = print
        self.output_dir = output_dir
        self.round = round

    def __call__(self, *args, **kwargs):
        nodes = kwargs["nodes"]
        scores = np.array(kwargs["scores"], dtype=object)
        labels = np.array(kwargs["labels"])

        for i, node in enumerate(nodes):
            print(node, f"accuracy: {accuracy(scores[i], labels, self.round) * 100}%")
