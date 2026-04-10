from graph import Graph
from graph import Node
from graph import Edge
from enum import Enum

class GraphRenderer(object):
    """ Base class for graph renderers. """
    def __init__(self, graph, id="graph"):
        self.graph = graph
        self.id = id
        self.width = graph.width or 800
        self.height = graph.height or 600
        self.weighted = graph.weighted or 1
        self.directed = graph.directed or True
        self.ipf = graph.ipf or 0.1
        self.frames = graph.frames or 25000

    @property
    def data(self):
        """ Yields a string of JSON with the graph's data. """
        s = self.graph.to_json()
        return s

    @property
    def script(self):
        """ Yields a string of JavaScript for the graph visualization. """
        s = """var nodes = {nodes}
             var links = {links}
             var data = {data}
             var simulation = new Simulation(data, graph)
             simulation.start()"""
        s = s.replace("\n", "").replace("\t", "")
        s = s.format(
            nodes=self.graph.to_json(),
            links=self.graph.to_json(),
            data=self.graph.to_json()
        )
        return s

    def _script(self):
        """ Yields a string of JavaScript to visualize the graph using a force-based spring layout. """
        return """<script type="text/canvas">
            (function() {
                var canvas = document.getElementById("%s")
                var graph = new Visio.Graph(canvas)
                graph.on("render")
                graph.on("click")
                graph.on("mouseover")
                graph.on("mouseout")
                graph.on("drag")
                graph.on("resize")
                graph.on("pan")
                graph.on("zoom")
                graph.on("clear")
                graph.on("add")
                graph.on("remove")
                graph.on("edit")
                graph.on("save")
                graph.on("load")
            }())
            % (self.id)
        """
        return """
            var nodes = %s
            var links = %s
            var graph = new Visio.Graph(document.getElementById("%s"))
            graph.nodes = nodes
            graph.links = links
            graph.on("click", function(node) {
                console.log("clicked", node)
                if (node.label) {
                    alert(node.label)
                }
            })
            graph.on("mouseover", function(node) {
                console.log("mouseover", node)
            })
            graph.on("mouseout", function(node) {
                console.log("mouseout", node)
            })
            graph.on("drag", function(node) {
                console.log("drag", node)
            })
            graph.on("resize", function(node) {
                console.log("resize", node)
            })
            graph.on("pan", function(node) {
                console.log("pan", node)
            })
            graph.on("zoom", function(node) {
                console.log("zoom", node)
            })
            graph.on("clear", function() {
                console.log("clear")
            })
            graph.on("add", function(node) {
                console.log("add", node)
            })
            graph.on("remove", function(node) {
                console.log("remove", node)
            })
            graph.on("edit", function(node) {
                console.log("edit", node)
            })
            graph.on("save", function() {
                console.log("save")
            })
            graph.on("load", function() {
                console.log("load")
            })
            graph.render()
        """

    @property
    def javascript(self):
        """ Yields a string with the path to the JavaScript files. """
        return self.graph.javascript or ""

    @property
    def stylesheet(self):
        """ Yields a string with the path to the stylesheet. """
        return self.graph.stylesheet or ""

    @property
    def title(self):
        """ Yields a string with the graph title. """
        return self.graph.title or "Graph"

    @property
    def html(self):
        """ Yields a string of HTML to visualize the graph using a force-based spring layout.
            The js parameter sets the path to graph.js and canvas.js.
        """
        js = self.javascript or ""
        if self.stylesheet == INLINE:
            css = self.style.replace("\n", "\n\t\t").rstrip("\t")
            css = "<style type=\"text/css\">\n\t\t%s\n\t</style>" % css
        elif self.stylesheet == DEFAULT:
            css = "<link rel=\"stylesheet\" href=\"style.css\" type=\"text/css\" media=\"screen\" />"
        elif self.stylesheet is not None:
            css = "<link rel=\"stylesheet\" href=\"%s\" type=\"text/css\" media=\"screen\" />" % self.stylesheet
        else:
            css = ""
        s = self._script()
        s = "".join(s)
        s = "\t" + s.replace("\n", "\n\t\t\t")
        s = s.rstrip()
        s = self._source % (
            self.title,
            css,
            js,
            js,
            self.id,
            self.width,
            self.height,
            s)
        return s

    def serialize(self, type=HTML):
        if type == HTML:
            return self.html
        if type == CANVAS:
            return self.canvas
        if type in (STYLE, CSS):
            return self.style
        if type == SCRIPT:
            return self.script
        if type == DATA:
            return self.data

    # Backwards compatibility.
    render = serialize

    def export(self, path, encoding="utf-8"):
        """ Generates a folder at the given path containing an index.html
            that visualizes the graph using the HTML5 <canvas> tag.
        """
        if os.path.exists(path):
            rmtree(path)
        os.mkdir(path)
        # Copy compressed graph.js + canvas.js (unless a custom path is given.)
        if self.javascript is None:
            for p, f in (("..", "canvas.js"), (".", "graph.js")):
                a = open(os.path.join(MODULE, p, f), "r")
                b = open(os.path.join(path, f), "w")
                b.write(minify(a.read()))
                b.close()
        # Create style.css.
        if self.stylesheet == DEFAULT:
            f = open(os.path.join(path, "style.css"), "w")
            f.write(self.style)
            f.close()
        # Create index.html.
        f = open(os.path.join(path, "index.html"), "w", encoding=encoding)
        f.write(self.html)
        f.close()

class Canvas(object):
    """ A HTML5 <canvas> element. """
    def __init__(self, path, graph=None, encoding="utf-8"):
        self.path = path
        self.graph = graph
        self.encoding = encoding
        self.render = self.render

    @property
    def render(self):
        return self.graph.render

    def render(self, path=None):
        """ Renders the graph to an HTML file at the given path. """
        if path is None:
            path = self.path or "graph.html"
        return GraphRenderer.render(self.graph, path, self.encoding)

    @property
    def to_html(self):
        """ Yields an HTML <canvas> element that renders the graph. """
        return self.render(self.path)

    @property
    def to_html(self):
        """ Yields an HTML <canvas> element that renders the graph. """
        return self.render(self.path)

    @property
    def canvas(self):
        """ Yields a string with the path to the HTML5 <canvas> element. """
        return self.path or "graph.html"

    @property
    def style(self):
        """ Yields a string with the path to the stylesheet. """
        return self.stylesheet or ""

    @property
    def stylesheet(self):
        """ Yields a string with the path to the stylesheet. """
        return self.stylesheet or ""

    @property
    def script(self):
        """ Yields a string of JavaScript to visualize the graph using a force-based spring layout. """
        return self.graph.script or ""

    def serialize(self, type=HTML):
        if type == HTML:
            return self.to_html
        if type == CANVAS:
            return self.canvas
        if type == DATA:
            return self.to_json

# Backwards compatibility.
from graph import Graph
graph, render = export, serialize