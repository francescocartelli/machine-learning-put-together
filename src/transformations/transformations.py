import numpy as np
from src.graphs.graphs import Transformation


class Stack(Transformation):
    def __init__(self, direction='v'):
        if direction != 'v' and direction != 'h':
            raise Exception(f"Stack vertically with 'v' or horizontally with 'h', receive {direction} as parameter")
        self.direction = direction

    def transform(self, *x):
        x = [x_[1] - x_[0] if len(x_.shape) > 1 else x_ for x_ in x]
        return np.vstack(x) if self.direction == 'v' else np.hstack

    def __str__(self):
        return "+"
