import matplotlib.pyplot as plt
import networkx as nx

from src.graphs.grids import Grid


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
    def __init__(self, node, node_id=0, inputs=None, label=None):
        self.node = node
        self.results = None
        self.inputs = inputs

        self.node_id = node_id
        self.level = self.__level()
        self.label = label

    def __level(self):
        levels = [i.level for i in self.inputs]
        return max(levels) + 1 if levels else 0

    def __call__(self):
        return self.results

    def __str__(self):
        return str(self.node) if self.label is None else self.label


class InputNode:
    def __init__(self, type=None):
        self.type = type

    def __str__(self):
        return f"{self.type}"


class Graph:
    def __init__(self):
        self.levels = [0]

        self.schedule = []
        self.add(InputNode, type='D', inputs=[])
        self.add(InputNode, type='L', inputs=[])

        self.x = self.schedule[0]
        self.y = self.schedule[1]

    def add(self, node_class, **kwargs):
        inputs = kwargs.pop("inputs")
        label = kwargs.pop("label", None)
        graph_node = GraphNode(node_class(**kwargs), len(self.schedule) + 1, inputs=inputs, label=label)

        self.schedule.append(graph_node)
        try:
            self.levels[graph_node.level] += 1
        except IndexError:
            self.levels.append(1)

        return graph_node

    def add_multiple(self, node_class, grid, **kwargs):
        inputs = kwargs.pop("inputs")
        labels = kwargs.pop("labels", None)

        graph_nodes = []
        for config in grid.configs:
            node = node_class(**config)
            graph_node = GraphNode(node, len(self.schedule) + 1, inputs=inputs, label=labels(node) if labels is not None else None)
            graph_nodes.append(graph_node)
            self.schedule.append(graph_node)
        try:
            self.levels[graph_nodes[0].level] += len(graph_nodes)
        except IndexError:
            self.levels.append(len(graph_nodes))

        return graph_nodes

    def fit(self, x, y):
        self.x.results = x
        self.y.results = y
        for n in self.schedule:
            if isinstance(n.node, Classifier):
                n.results = n.node.train_transform(*(i() for i in n.inputs))
            elif isinstance(n.node, Node):
                n.results = n.node.fit_transform(n.inputs[0]())
            elif isinstance(n.node, Transformation):
                n.results = n.node.transform(*(i() for i in n.inputs))

    def transform(self, x):
        self.x.results = x
        for n in self.schedule:
            if isinstance(n.node, Classifier):
                n.results = n.node.transform([g() for g in n.inputs][0])
            elif isinstance(n.node, Transformation):
                n.results = n.node.transform(*(i() for i in n.inputs))

    def output(self, y):
        for n in self.schedule:
            if isinstance(n.node, Printer):
                scores = [i() for i in n.inputs]
                nodes = [i.__str__() for i in n.inputs]
                n.node(nodes=nodes, scores=scores, labels=y)

    def display(self):
        node_labels = {}
        node_colors = []
        lev = [l/2 for l in self.levels]

        g = nx.Graph()
        for i, n in enumerate(self.schedule):
            node_colors.append('g' if i == 0 or i == 1 else 'b')
            if isinstance(n.node, Classifier):
                color = "#2471A3"
            elif isinstance(n.node, Node):
                color = "#196F3D"
            elif isinstance(n.node, Transformation):
                color = "#196F3D"
            elif isinstance(n.node, InputNode):
                color = "#7D3C98"
            else:
                color = "#2E4053"
            g.add_node(n.node_id, pos=(n.level, lev[n.level]), color=color)
            lev[n.level] -= 1
            for k, j in enumerate(n.inputs):
                color = '#BFC9CA' if isinstance(n.node, Classifier) and k == 1 else '#5F6A6A'
                style = 'dashed' if isinstance(n.node, Classifier) and k == 1 else 'solid'
                g.add_edge(j.node_id, n.node_id, color=color, style=style)
            node_labels[n.node_id] = str(n)
        options = {
            "width": 1,
            "labels": node_labels,
            "with_labels": True,
            "arrows": True,
            "node_color": [n[1]['color'] for n in g.nodes(data=True)],
            "node_size": 1000,
            "edge_color": [g[u][v]['color'] for u, v in g.edges],
            "style": [g[u][v]['style'] for u, v in g.edges],
            "connectionstyle": 'arc3,rad=0.2',
            "font_size": 8,
            "font_color": "#F0F0F0"
        }
        nx.draw(g, nx.get_node_attributes(g, 'pos'), **options)
        plt.show()


