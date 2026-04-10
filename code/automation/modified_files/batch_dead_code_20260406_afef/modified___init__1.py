from graphlib import export, serialize

# Export graph as HTML
html = export(my_graph, type='html')

# Export graph to file
export(my_graph, '/path/to/graph.html', type='html')

# Export graph as GraphML
graphml = serialize(my_graph, type='graphml')

# Render graph to canvas
rendered = serialize(my_graph, type='canvas')