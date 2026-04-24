from imgaug import augmenters
from imgaug.augmentables import BoundingBoxesOnImage, BoundingBox

# Create bounding boxes
bbox = BoundingBox(100, 100, 200, 200, label="Example")

# Use on image
bbox_list = BoundingBoxesOnImage([bbox], shape=(512, 512, 3))

# Draw labels on image
image = np.zeros((512, 512, 3), dtype=np.uint8)
drawer = _LabelOnImageDrawer(color=(255, 0, 0))
drawer.draw_on_image(image, bbox_list)