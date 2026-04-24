import os
import sys
sys.path.insert(0, os.path.join("..", ".."))

from pattern.graph import Graph
from random import choice, random

g = Graph()
# Random nodes.
for i in range(50):
    g.add_node(i)
# Random edges.
for i in range(75):
    node1 = choice(g.nodes)
    node2 = choice(g.nodes)
    g.add_edge(node1, node2,
               weight = random())

g.prune(0)

# This node's label is different from its id.
g[1].text.string = "home"

# By default, Graph.export() exports to HTML,
# but if we give it a filename that ends in .graphml it will export to GraphML.
g.export(os.path.join(os.path.dirname(__file__), "test.graphml"))