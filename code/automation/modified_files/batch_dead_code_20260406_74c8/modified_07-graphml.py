from graphlib import Graph
import graphml
import os

# Initialize the graph
g = Graph()

# Add nodes (using standard naming conventions for clarity)
g.add_node('A', label='Node A')
g.add_node('B', label='Node B')
g.add_node('C', label='Node C')

# Add edges
g.add_edge(('A', 'B'))
g.add_edge(('B', 'C'))
g.add_edge(('A', 'C'))

# Add properties
g['A']['data'] = 'properties'
g['B']['data'] = 'more'
g['C']['data'] = 'properties'

# Save the graph to a file
filepath = './test.graphml'
# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Write the graph to GraphML
graphml.write(g, filepath)

print(f"Graph saved to {filepath}")