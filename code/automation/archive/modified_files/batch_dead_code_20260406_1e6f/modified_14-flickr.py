import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pattern.web import Flickr, extension, RELEVANCY, LATEST, INTERESTING, SMALL, MEDIUM, LARGE
from io import open

engine = Flickr(license=None)
q = "duracell bunny"
results = engine.search(q, size=MEDIUM, sort=RELEVANCY, cached=False)
for img in results:
    print(img.text)
    print(img.author)
    print("")

img = results[0]
data = img.download()
path = q.replace(" ", "_") + extension(img.url)
f = open(path, "wb")
f.write(data)
f.close()
print("Download: %s" % img.url)
print("Saved as: %s" % path)