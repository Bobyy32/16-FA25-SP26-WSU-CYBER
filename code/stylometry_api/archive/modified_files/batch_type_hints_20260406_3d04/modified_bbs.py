import imgaug as ia

# Draw a bounding box with a label
bbox = ia.BoundingBoxOnImage(0.1, 0.2, 0.3, 0.4, label="person")
image = np.zeros((300, 300, 3), dtype=np.uint8)
drawer = ia._LabelOnImageDrawer(color=(255, 255, 255), color_bg=(0, 0, 255))
image = drawer.draw_on_image(image, bbox)