bb = BoundingBoxesOnImage(
    [BoundingBox(0, 0, 100, 100, 1)],
    shape=(100, 100)
)

drawer = _LabelOnImageDrawer(color=(0, 0, 255))
image = np.zeros((100, 100, 3), dtype=np.uint8)
drawn_image = drawer.draw_on_image(image, bb[0])