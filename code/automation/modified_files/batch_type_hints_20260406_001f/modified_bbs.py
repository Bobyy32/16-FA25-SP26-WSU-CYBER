from imgaug.augmentables.bbs import BoundingBoxesOnImage

bboxes = BoundingBoxesOnImage([
    BoundingBox(10, 20, 100, 50, label="Object1")
], shape=(100, 100, 3))

# Access bounding box at index 0
bb = bboxes[0]
print(bb.x1, bb.y1, bb.x2, bb.y2)

# Iterate over boxes
for bb in bboxes:
    print(f"Label: {bb.label}")

# Get number of bounding boxes
print(len(bboxes))