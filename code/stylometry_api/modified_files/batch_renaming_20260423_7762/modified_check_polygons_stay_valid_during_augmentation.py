from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main_processor():
    nb_checked = 0

    augs = iaa.SomeOf((1, None), [
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
        polygon = generate_polygon_processor(height, width, nb_checked)
        psoi = PolygonsOnImage([polygon], shape=(height, width, 3))
        psoi_aug = augs.augment_polygons(psoi)

        if not polygon.is_valid or not psoi_aug.polygons[0].is_valid:
            print("polygon:     ", polygon, polygon.is_valid)
            print("polygon_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        assert polygon.is_valid
        assert psoi_aug.polygons[0].is_valid

        nb_checked += 1
        if nb_checked % 100 == 0:
            print("Checked %d..." % (nb_checked,))
        if nb_checked > 100000:
            break


def generate_polygon_processor(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 0] *= width
    coords[:, 1] *= height
    polygon = Polygon(coords)
    if polygon.is_valid:
        return polygon

    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return generate_polygon_processor(height, width, new_seed)


if __name__ == "__main__":
    main_processor()