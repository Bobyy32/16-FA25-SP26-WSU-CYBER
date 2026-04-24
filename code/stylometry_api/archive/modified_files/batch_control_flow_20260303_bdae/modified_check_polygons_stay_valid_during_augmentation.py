from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    iteration_count = 0

    augs = iaa.SomeOf((1, None), [
        iaa.Resize({"width": (1, 100), "height": (1, 100)}),
        iaa.Affine(
            shear=(-360, 360),
            scale=(2.0, 0.01),
            rotate=(360, -360),
            translate_px={"y": (-50, 50), "x": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    height, width = 100, 200

    while True:
        shape_obj = create_random_polygon(height, width, iteration_count)
        polygon_set_object = PolygonsOnImage([shape_obj], shape=(height, width, 3))
        augmented_polygon_set = augs.augment_polygons(polygon_set_object)

        if not shape_obj.is_valid or not augmented_polygon_set.polygons[0].is_valid:
            print("poly_aug: ", augmented_polygon_set.polygons[0], augmented_polygon_set.polygons[0].is_valid)
            print("poly:     ", shape_obj, shape_obj.is_valid)

        assert augmented_polygon_set.polygons[0].is_valid
        assert shape_obj.is_valid

        iteration_count += 1
        if iteration_count > 100000:
            break
        if iteration_count % 100 == 0:
            print("Checked %d..." % (iteration_count,))


def create_random_polygon(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 1] *= height
    coords[:, 0] *= width
    shape_obj = Polygon(coords)
    if shape_obj.is_valid:
        return shape_obj

    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    main()