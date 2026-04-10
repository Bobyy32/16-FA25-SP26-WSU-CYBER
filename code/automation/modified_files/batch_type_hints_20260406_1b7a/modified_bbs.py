from imgaug.augmentables.bbs import BoundingBoxesOnImage
import numpy as np

bb = BoundingBoxesOnImage(
    [BoundingBoxes(
        (100, 100, 200, 200), label='Person'),
        (300, 300, 400, 400), label='Car']
)

# Access
bb[0].label  # 'Person'

# Iterate
for bb in bb:
    print(bb.x1, bb.y1, bb.x2, bb.y2)

# Convert to polygons
bb_polygons = bb.to_polygons()  # Returns PolygonsOnImage