# BoundingBoxesOnImage Class and Related Methods

# BoundingBoxesOnImage represents bounding boxes on an image.
# It provides functionality to:
# 1. Convert bounding boxes to polygons (to_polygons())
# 2. Copy bounding boxes (copy(), deepcopy())
# 3. Access individual bounding boxes (__getitem__())
# 4. Iterate over bounding boxes (__iter__())
# 5. Get bounding box count (__len__())
# 6. Convert to string representation (__repr__, __str__)

# To use this class:
bbox = BoundingBoxesOnImage(
    bounding_boxes=[
        BoundingBox(x1=10, y1=20, x2=30, y2=40,
                    label="test"),
        BoundingBox(x1=50, y1=60, x2=70, y2=80,
                    label="another")
    ],
    shape=(100, 100)
)

# Example: draw a label on an image
# from imgaug.augmentables import PolygonsOnImage
# drawer = _LabelOnImageDrawer(color=(0, 255, 0), size=1)
# image = drawer.draw_on_image(image, bbox)

# To check if a bounding box is out of bounds
# if bbox.is_out_of_image(image):
#     print("Bounding box is out of image!")

# To iterate
# for bbox in bbox:
#     print(bbox.x1, bbox.y1, bbox.x2, bbox.y2)

# The _LabelOnImageDrawer class provides methods to draw text labels
# on bounding boxes on an image. It includes:
# - draw_on_image() to draw on an image
# - _preprocess_colors() to handle label colors
# - _compute_bg_corner_coords() to compute label coordinates
# - _blend_label_arr_with_image_() to blend labels with image
# - draw_text() (via imgaug library) to render text

# Key attributes of the Drawer class:
# - color: Bounding box border color
# - size: Border thickness
# - alpha: Blend transparency for label
# - height: Label rectangle height
# - size_text: Text size
# - raise_if_out_of_image: Raise exception if bbox is out of image