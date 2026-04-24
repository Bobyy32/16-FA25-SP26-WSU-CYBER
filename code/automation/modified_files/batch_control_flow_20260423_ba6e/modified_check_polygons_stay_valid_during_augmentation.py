from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


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

        match poly.is_valid, psoi_aug.polygons[0].is_valid:
            case (False, False):
                print("poly:     ", poly, poly.is_valid)
                print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)
            case (True, False):
                print("poly:     ", poly, poly.is_valid)
                print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)
            case (False, True):
                print("poly:     ", poly, poly.is_valid)
                print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)
            case (True, True):
                continue
        else:
            break

        assert poly.is_valid
        assert psoi_aug.polygons[0].is_valid

        nb_checked += 1
        match nb_checked % 100:
            case 0:
                if nb_checked == 100000:
                    break
                else:
                    print("Checked %d..." % nb_checked)


def create_random_polygon(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)
    coords[:, 0] *= width
    coords[:, 1] *= height
    poly = Polygon(coords)

    match poly.is_valid:
        case True:
            return poly
        case False:
            new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
            return create_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    main()