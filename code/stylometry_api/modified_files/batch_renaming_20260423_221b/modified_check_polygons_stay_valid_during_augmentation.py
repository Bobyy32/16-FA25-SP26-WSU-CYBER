from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def data_process():
    check_count = 0

    data_augmenters = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    height, width = 100, 200

    while True:
        data = data_create_random_poly(height, width, check_count)
        data_polys = PolygonsOnImage([data], shape=(height, width, 3))
        data_polys_aug = data_augmenters.augment_polygons(data_polys)

        if not data.is_valid or not data_polys_aug.polygons[0].is_valid:
            print("data:     ", data, data.is_valid)
            print("data_polys_aug: ", data_polys_aug.polygons[0], data_polys_aug.polygons[0].is_valid)

        assert data.is_valid
        assert data_polys_aug.polygons[0].is_valid

        check_count += 1
        if check_count % 100 == 0:
            print("Checked %d..." % (check_count,))
        if check_count > 100000:
            break


def data_create_random_poly(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 0] *= width
    coords[:, 1] *= height
    data = Polygon(coords)
    if data.is_valid:
        return data

    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return data_create_random_poly(height, width, new_seed)


if __name__ == "__main__":
    data_process()