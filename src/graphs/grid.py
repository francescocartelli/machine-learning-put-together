import itertools


class Grid:
    def __init__(self, *args, **kwargs):
        keys, values = zip(*kwargs.items())

        self.configs = [dict(zip(keys, v)) for v in itertools.product(*values)]