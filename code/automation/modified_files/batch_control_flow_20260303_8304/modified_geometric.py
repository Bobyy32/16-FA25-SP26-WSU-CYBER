psois = [bbsoi.to_polygons_on_image() for bbsoi in batch.bounding_boxes]
psois = [psoi.subdivide_(2) for psoi in psois]