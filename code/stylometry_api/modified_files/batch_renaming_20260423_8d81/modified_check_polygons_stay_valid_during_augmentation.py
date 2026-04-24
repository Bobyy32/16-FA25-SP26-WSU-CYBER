from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main_():
    nb_checked_ = 0

    augs_ = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    height_, width_ = 100, 200

    while True:
        poly_ = create_random_polygon_(height_, width_, nb_checked_)
        psoi_ = PolygonsOnImage([poly_], shape=(height_, width_, 3))
        psoi_aug_ = augs_.augment_polygons_(psoi_)

        if not poly_.is_valid or not psoi_aug_.polygons_[0].is_valid:
            print("poly:     ", poly_, poly_.is_valid)
            print("poly_aug: ", psoi_aug_.polygons_[0], psoi_aug_.polygons_[0].is_valid)

        assert poly_.is_valid
        assert psoi_aug_.polygons_[0].is_valid

        nb_checked_ += 1
        if nb_checked_ % 100 == 0:
            print("Checked %d..." % (nb_checked_,))
        if nb_checked_ > 100000:
            break


def create_random_polygon_(height_, width_, seed_):
    rs_ = np.random.RandomState_(seed_)
    nb_points_ = rs_.randint_(3, 50)
    coords_ = rs_.rand_(nb_points_, 2)
    coords_ = (coords_ * 2 - 0.5)  # allow coords outside of the image plane
    coords_[:, 0] *= width_
    coords_[:, 1] *= height_
    poly_ = Polygon_(coords_)
    if poly_.is_valid:
        return poly_

    new_seed_ = rs_.randint_(ia.SEED_MIN_VALUE_, ia.SEED_MAX_VALUE_)
    return create_random_polygon_(height_, width_, new_seed_)


if __name__ == "__main__":
    main_()