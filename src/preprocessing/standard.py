import numpy as np

from src.graphs import *


class Standardization(Node):
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, x):
        self.mu = x.mean(axis=1).reshape(x.shape[0], 1)
        self.sigma = x.std(axis=1).reshape(x.shape[0], 1)

    def transform(self, x):
        return (x - self.mu) / self.sigma

    def __str__(self):
        return f"Std"
