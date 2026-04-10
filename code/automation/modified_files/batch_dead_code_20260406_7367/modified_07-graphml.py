from pattern.graph import Graph
from random import choice, random
import os
import sys

sys.path.insert(0, os.path.join("..", ".."))

g = Graph()
for i in range(50):
    g.add_node(i)
for i in range(75):
    node1 = choice(g.nodes)
    node2 = choice(g.nodes)
    g.add_edge(node1, node2, weight = random())

g.prune(0)
g[1].text.string = "home"
g.export(os.path.join(os.path.dirname(__file__), "test.graphml"))