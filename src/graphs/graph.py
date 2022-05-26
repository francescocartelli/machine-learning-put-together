class Printer:
    def __call__(self, *args, **kwargs):
        pass


class Transformation:
    def transform(self, *x):
        pass

    def __str__(self):
        pass


class Node(Transformation):
    def fit(self, x):
        pass

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class Classifier(Node):
    def train(self, x, y):
        pass

    def train_transform(self, x, y):
        self.train(x, y)
        return self.transform(x)


class GraphNode:
    def __init__(self, node, inputs=None):
        self.node = node
        self.results = None
        self.inputs = inputs
        self.steps = ""

    def o(self):
        return self.results

    def __str__(self):
        return self.steps


class InputNode:
    def __init__(self):
        pass


class Graph:
    def __init__(self):
        self.schedule = []
        self.x, self.y = None, None

    def add(self, node_class, **kwargs):
        inputs = kwargs["inputs"]
        del kwargs["inputs"]
        graph_node = GraphNode(node_class(**kwargs), inputs)
        self.schedule.append(graph_node)

        return graph_node

    def fit(self, x, y):
        self.x, self.y = x, y
        for n in self.schedule:
            if isinstance(n.node, Classifier):
                inp = [g() for g in n.inputs]
                n.node.train(inp[0], inp[1])
                n.results = n.node.transform(inp[0])
            elif isinstance(n.node, Node):
                n.results = n.node.fit_transform(n.inputs())
            elif isinstance(n.node, Transformation):
                n.results = n.node.transform(n.inputs())

    def transform(self, x):
        self.x = x
        for n in self.schedule:
            if isinstance(n.node, Classifier):
                n.results = n.node.transform([g() for g in n.inputs][0])
            elif isinstance(n.node, Transformation) or isinstance(n.node, Node):
                n.results = n.node.transform(n.inputs())

    def output(self, y):
        for n in self.schedule:
            if isinstance(n.node, Printer):
                scores = [i.o() for i in n.inputs]
                nodes = [i.node.__str__() for i in n.inputs]
                n.node(nodes=nodes, scores=scores, labels=y)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __call__(self, i=None):
        return self.schedule[0], i if i is not None else 0


