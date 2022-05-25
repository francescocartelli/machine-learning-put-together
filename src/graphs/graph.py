class Transformation:
    def transform(self, x):
        pass

    def __str__(self):
        pass


class Node(Transformation):
    def fit(self, x):
        pass

    def fit_transform(self, x, y):
        pass


class Classifier(Node):
    def train(self, x, y):
        pass


class GraphNode:
    def __init__(self, node, inputs=None):
        self.node = node
        self.results = None
        self.inputs = inputs
        self.steps = ""

    def o(self):
        return self.results


class InputNode:
    def __init__(self):
        pass


class Graph:
    def __init__(self):
        self.schedule = []
        self.x, self.y = None, None

    def add_node(self, node_class, *args, inputs=None):
        graph_node = GraphNode(node_class(*args), inputs)
        self.schedule.append(graph_node)

        return graph_node

    def fit(self, x, y):
        self.x, self.y = x, y
        for n in self.schedule:
            if isinstance(n.node, Classifier):
                inp = [g() for g in n.inputs]
                n.node.train(inp[0], inp[1])
                n.results = n.node.transform(inp[0])
            elif isinstance(n.node, Transformation) or isinstance(n.node, Node):
                n.results = n.node.transform(n.inputs())

    def transform(self, x):
        pass

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __call__(self, i=None):
        return self.schedule[0], i if i is not None else 0


