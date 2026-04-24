import os
import sys
from pattern.graph import Graph, WEIGHT, CENTRALITY, DEGREE, DEFAULT
from random import choice, random

g = Graph()
for i in range(50):
    g.add_node(i)
for i in range(75):
    node1 = choice(g.nodes)
    node2 = choice(g.nodes)
    g.add_edge(node1, node2, weight=random())

g.prune(0)
g[1].text.string = "home"
g.export(os.path.join(os.path.dirname(__file__), "test.graphml"))