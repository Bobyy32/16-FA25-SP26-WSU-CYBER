import imgaug.augmentables.bbs

boxes = imgaug.augmentables.bbs.BoundingBoxesOnImage([
    imgaug.augmentables.bbs.BoundingBox([10, 20], [100, 50], label='dog'),
    imgaug.augmentables.bbs.BoundingBox([30, 40], [120, 60], label='cat'),
], shape=(256, 256, 3))

# Print number of boxes
print(len(boxes))

# Get a specific box
box = boxes[0]
print(box.label)

# Copy and draw labels
boxes_copy = boxes.deepcopy()

# Draw labels on image
image = np.zeros((256, 256, 3), dtype=np.uint8)
drawer = _LabelOnImageDrawer()
image = boxes_copy.draw_labels_on_image(image)