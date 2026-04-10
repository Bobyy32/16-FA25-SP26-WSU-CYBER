import imgaug
from imgaug.augmentables.bbs import BoundingBoxesOnImage

# Define bounding boxes
box = [
    BoundingBox(20, 20, 50, 50, label='object1'),
    BoundingBox(70, 10, 120, 50, label='object2')
]

# Create BoundingBoxesOnImage object
image = (255,) * 1000000
bbox_on_img = BoundingBoxesOnImage(box, shape=(100, 100))

# Draw labels on image
bbox_on_img.draw_on_image(image)