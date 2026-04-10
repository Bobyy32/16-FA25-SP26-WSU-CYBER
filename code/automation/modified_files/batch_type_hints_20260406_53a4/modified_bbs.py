bbs = BoundingBoxesOnImage(
      [(0, 0, 100, 100)],
      shape=(200, 200, 3)
  )
  polys = bbs.convert_to_polygons()  # Returns PolygonsOnImage