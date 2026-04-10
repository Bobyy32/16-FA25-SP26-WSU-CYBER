# Complete HTML canvas rendering script
from __future__ import absolute_import

# Constants for stylesheet handling
INLINE = "inline"
DEFAULT = "/path/to/style.css"  # or None
STYLE = "style"
CANVAS = "canvas"
HTML = "html"
GRAPHML = "graphml"
SCRIPT = "script"
DATA = "data"

class HTMLCanvasRenderer:
    def __init__(self, graph, id="graph", width=800, height=600,
                 stylesheet=DEFAULT, title="Graph", javascript=None):
        self.graph = graph
        self.id = id
        self.width = width
        self.height = height
        self.stylesheet = stylesheet
        self.title = title
        self.javascript = javascript
        
        self.script = self._script()
        self.data = self._data()
        
    def _script(self):
        """ Returns the JavaScript code for the canvas rendering. """
        # This is the full force-based layout script
        return "\t\t".join([
            "graph.setOptions({\n",
                "\t\t\tnodes:\n",
                "\t\t\t\t\t%s\n",
                "\t\t\t\t,\n",
                "\t\t\t\tlinks:\n",
                "\t\t\t\t\t%s\n",
            "});\n",
            "graph.start({\n",
                "\t\t\tpower:\n",
                "\t\t\t\t%s,\n",
                "\t\t\t\tinterpolate:\n",
                "\t\t\t\t%s,\n",
                "\t\t\t\tweighted:\n",
                "\t\t\t\t%s,\n",
                "\t\t\t\tdirected:\n",
                "\t\t\t\t%s,\n",
            "});\n",
            "graph.on(\"stop\",\n",
                "\t\t\tfunction()\n",
            "{\n",
            "\t\tdrag:\n",
            "\t\t\tfunction(d)\n",
            "\t\t\t{\n",
            "\t\t\t\td.drag(canvas.mouse);\n",
            "\t\t\t}\n",
            "\t});\n"
        ]) % (
            self._data(),  # nodes
            self._data(),  # edges
            int(self.frames),  # power
            int(self.ipf),  # interpolate
            str(self.weighted).lower(),  # weighted
            str(self.directed).lower())  # directed
        
    @property
    def canvas(self):
        """ Yields a string of HTML with a <div id="graph"> containing a <script type="text/canvas">. """
        s = [
            "<div id=\"%s\" style=\"width:%spx; height:%spx;\">\n" % (self.id, self.width, self.height),
                "\t<script type=\"text/canvas\">\n",
                "\t\t%s\n" % self.script.replace("\n", "\n\t\t"),
                "\t</script>\n",
            "</div>"
        ]
        return "".join(s)
    
    @property
    def style(self):
        """ Yields a string of CSS for <div id="graph">. """
        return \
            "body { font: 11px sans-serif; }\n" \
            "a { color: dodgerblue; }\n" \
            "#%s canvas { }\n" \
            "#%s .node-label { font-size: 11px; }\n" \
            "#%s {\n" \
                "\tdisplay: inline-block;\n" \
                "\tposition: relative;\n" \
                "\toverflow: hidden;\n" \
                "\tborder: 1px solid #ccc;\n" \
            "}" % (self.id, self.id, self.id)
    
    @property
    def html(self):
        """ Yields a string of HTML to visualize the graph using a force-based spring layout. """
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
        """ Generates a folder at the given path containing an index.html """
        if os.path.exists(path):
            rmtree(path)
        os.mkdir(path)
        # Copy compressed graph.js + canvas.js
        if self.javascript is None:
            for p, f in (("..", "canvas.js"), (".", "graph.js")):
                a = open(os.path.join(MODULE, p, f), "r")
                b = open(os.path.join(path, f), "w")
                b.write(a.read())
                b.close()
        # Copy compressed graph.css + script.css
        if self.stylesheet is None:
            for f in ("style.css", "script.css"):
                b = open(os.path.join(MODULE, f), "r")
                a = open(os.path.join(path, f), "w")
                a.write(b.read())
                a.close()
        # Copy compressed graph.svg + graph.svg.js + graph.html
        a = open(os.path.join(path, "index.html"), "wb")
        b = open(os.path.join(MODULE, "index.html"), "r")
        a.write(b.read())
        a.close()
        # Copy compressed graph.js + graph.html
        a = open(os.path.join(path, "graph.html"), "wb")
        b = open(os.path.join(MODULE, "graph.html"), "r")
        a.write(b.read())
        a.close()