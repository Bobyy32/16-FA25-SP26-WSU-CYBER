# Render as HTML
html = graph.serialize(type=HTML, title="My Graph")

# Export to folder
graph.export("/output/graphs")

# Export to GraphML
xml = graph.serialize(type=GRAPHML)

# Export as GraphML file
graph.export("/output/mygraph.graphml", directed=True)