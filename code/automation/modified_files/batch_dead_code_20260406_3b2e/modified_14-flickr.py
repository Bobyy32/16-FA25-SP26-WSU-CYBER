from builtins import str

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pattern.web import Flickr, extension
from pattern.web import RELEVANCY, LATEST, INTERESTING
from pattern.web import SMALL, MEDIUM, LARGE

# Cache results from Flickr locally with cached=True parameter.
# Obtain your own license key at http://www.flickr.com/services/api/
engine = Flickr(license=None)

q = "duracell bunny"
results = engine.search(q, size=MEDIUM, sort=RELEVANCY, cached=False)
for img in results:
    # Retrieving the actual image URL executes a query.
    print(img.text)
    print(img.author)
    print("")

# Download and save one of the images:
img = results[0]
data = img.download()
path = q.replace(" ", "_") + extension(img.url)
f = open(path, "wb")
f.write(data)
f.close()
print("Download: %s" % img.url)
print("Saved as: %s" % path)