bbox = BoundingBoxOnImage(x1, y1, x2, y2, label='person')
bbox_on_image = BoundingBoxesOnImage([bbox], shape=(image_height, image_width))
bbox_on_image.draw_on_image(image)