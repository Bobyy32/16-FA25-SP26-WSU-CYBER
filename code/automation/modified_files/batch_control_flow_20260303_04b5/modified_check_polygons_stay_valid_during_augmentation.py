from __future__ import division, print_function
import numpy as np

import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.polys import PolygonsOnImage, Polygon


def main():
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
        poly = create_random_polygon(height, width, nb_checked)
        psoi = PolygonsOnImage([poly], shape=(height, width, 3))
        psoi_aug = augs.augment_polygons(psoi)

        if poly.is_valid and psoi_aug.polygons[0].is_valid:
            pass
        else:
            print("poly:     ", poly, poly.is_valid)
            print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        assert poly.is_valid
        assert psoi_aug.polygons[0].is_valid

        nb_checked += 1
        if nb_checked % 100 == 0:
            print("Checked %d..." % (nb_checked,))
        if nb_checked > 100000:
            break


def create_random_polygon(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)
    coords[:, 1] *= height
    coords[:, 0] *= width

    poly = Polygon(coords)
    if not poly.is_valid:
        return create_random_polygon(height, width, seed)

    return poly


if __name__ == "__main__":
    main()