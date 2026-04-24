from imgaug.augmentables.bbs import BoundingBoxesOnImage

# Create bounding boxes with labels and assign to an image
bbs = BoundingBoxesOnImage([
    (100, 200, 200, 250, 1),
    (150, 100, 250, 200, 2)
])

# Convert to polygons for use with polygons containers
polygons = bbs.to_polygons()

# Access or iterate over bounding boxes
print(bbs[0])  # First bounding box
print(list(bbs))  # Iterate over all boxes