from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def run():
    count = 0

    augmentation_pipeline = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    canvas_h, canvas_w = 100, 200

    while True:
        shape = create_polygon(canvas_h, canvas_w, count)
        poly_set = PolygonsOnImage([shape], shape=(canvas_h, canvas_w, 3))
        aug_shapes = augmentation_pipeline.augment_polygons(poly_set)

        if not shape.is_valid or not aug_shapes.polygons[0].is_valid:
            print("original_shape:  ", shape, shape.is_valid)
            print("augmented_shape: ", aug_shapes.polygons[0], aug_shapes.polygons[0].is_valid)

        assert shape.is_valid
        assert aug_shapes.polygons[0].is_valid

        count += 1
        if count % 100 == 0:
            print("Processed %d..." % (count,))
        if count > 100000:
            break


def create_polygon(canvas_h, canvas_w, seed):
    rs = np.random.RandomState(seed)
    point_count = rs.randint(3, 50)
    points = rs.rand(point_count, 2)
    points = (points * 2 - 0.5)  # allow points outside of the image plane
    points[:, 0] *= canvas_w
    points[:, 1] *= canvas_h
    polygon = Polygon(points)
    if polygon.is_valid:
        return polygon

    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_polygon(canvas_h, canvas_w, new_seed)


if __name__ == "__main__":
    run()