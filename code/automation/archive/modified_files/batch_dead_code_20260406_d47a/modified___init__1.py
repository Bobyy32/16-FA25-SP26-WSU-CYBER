class HTMLCanvasRenderer(GraphRenderer):
    """
    Renders graph as an HTML canvas with force-based spring layout.
    """

    def __init__(self, graph, **kwargs):
        self.id = kwargs.get("id", "graph")
        self.title = kwargs.get("title", "")
        self.width = kwargs.get("width", 600)
        self.height = kwargs.get("height", 400)
        self.fps = kwargs.get("fps", 30)
        self.weighted = kwargs.get("weighted", False)
        self.directed = kwargs.get("directed", False)
        self.ipf = kwargs.get("ipf", 10)
        self.frames = kwargs.get("frames", 1000)
        self.stylesheet = kwargs.get("stylesheet", HTML_CANVAS)
        self.javascript = kwargs.get("javascript")
        self.graph = graph

    @property
    def script(self):
        """ Generates JavaScript to visualize graph. """
        s = [
            "d3.v4.forceSimulation(this).node(this.getElementsByTagName('div'))",
            ".on('tick', function() { this.tick(); })",
            ".nodes(this.getElementsByClassName('node'))",
            ".links(this.getElementsByClassName('link'))",
            ".force('charge', d3.forceCharge().strength(%s))",
            ".force('link', d3.forceLink().distance(%s))",
            ".force('collide', d3.forceCollide().radius(%s))",
            ".on('tick', function() { this.tick(); })",
            ".on('tick', function() { this.tick(); })",
            ".on('end', function() {",
            "    var i = 0;",
            "    d3.select(this).selectAll('.node').each(function(d) {",
            "        var t = %s + d.text;",
            "        if (d3.select(this).select('.node-label').empty()) {",
            "            var t = document.createElement('div');",
            "            t.setAttribute('class', 'node-label');",
            "            t.textContent = d.text;",
            "            this.appendChild(t);",
            "            i++;",
            "        }",
            "        d3.select(this).select('.node-label').text(t);",
            "    });",
            "    d3.select('#%s').append('svg');",
            "    d3.select('#%s svg').selectAll('.link')",
            "        .data(d3.select('#%s').selectAll('.link').nodes())",
            "        .enter().append('line').attr('class', 'link');",
            "    d3.select('#%s svg').selectAll('.node')",
            "        .data(d3.select('#%s').selectAll('.node').nodes())",
            "        .enter().append('circle').attr('class', 'node');",
            "    d3.select('#%s svg').append('text')",
            "        .attr('class', 'node-text');",
            "});",
        ]
        return "\n".join(s)

    @property
    def canvas(self):
        """ Generates HTML canvas element. """
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
        """ Generates CSS style for graph. """
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
        """ Generates complete HTML for visualization. """
        js = self.javascript or ""
        if self.stylesheet == HTML_CANVAS:
            css = self.style.replace("\n", "\n\t\t").rstrip("\t")
            css = "<style type=\"text/css\">\n\t\t%s\n\t</style>" % css
        elif self.stylesheet == HTML_STYLE:
            css = "<link rel=\"stylesheet\" href=\"style.css\" type=\"text/css\" media=\"screen\" />"
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
        """ Generates output string based on requested type. """
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

    def export(self, path, encoding="utf-8"):
        """ Exports graph to a folder with index.html, canvas.js, graph.js, style.css. """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        if self.javascript is None:
            for p, f in (("..", "canvas.js"), (".", "graph.js")):
                a = open(os.path.join(MODULE, p, f), "r")
                b = open(os.path.join(path, f), "w")
                b.write(minify(a.read()))
                b.close()
        if self.stylesheet == HTML_STYLE:
            f = open(os.path.join(path, "style.css"), "w")
            f.write(self.style)
            f.close()
        f = open(os.path.join(path, "index.html"), "w", encoding=encoding)
        f.write(self.html)
        f.close()