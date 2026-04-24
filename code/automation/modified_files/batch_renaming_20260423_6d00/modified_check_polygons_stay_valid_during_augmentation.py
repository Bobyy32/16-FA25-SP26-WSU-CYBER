from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def _main_():
    _nb_checked_ = 0

    _augs_ = iaa.SomeOf((1, None), [
        iaa.Resize({"_height_": (1, 100), "_width_": (1, 100)}),
        iaa.Affine(
            _scale_=(0.01, 2.0),
            _rotate_=(-360, 360),
            _shear_=(-360, 360),
            _translate_px_={"_x_": (-50, 50), "_y_": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    _height_, _width_ = 100, 200

    while True:
        _poly_ = _create_random_polygon_(_height_, _width_, _nb_checked_)
        _psoi_ = PolygonsOnImage([_poly_], shape=(_height_, _width_, 3))
        _psoi_aug_ = _augs_._augment_polygons_(_psoi_)

        if not _poly_.is_valid or not _psoi_aug_._polygons_[0].is_valid:
            print("_poly_:     ", _poly_, _poly_.is_valid)
            print("_poly_aug: ", _psoi_aug_._polygons_[0], _psoi_aug_._polygons_[0].is_valid)

        assert _poly_.is_valid
        assert _psoi_aug_._polygons_[0].is_valid

        _nb_checked_ += 1
        if _nb_checked_ % 100 == 0:
            print("Checked %d..." % (_nb_checked_))
        if _nb_checked_ > 100000:
            break


def _create_random_polygon_(_height_, _width_, _seed_):
    _rs_ = np.random.RandomState(_seed_)
    _nb_points_ = _rs_.randint(3, 50)
    _coords_ = _rs_.rand(_nb_points_, 2)
    _coords_ = (_coords_ * 2 - 0.5)  # allow coords outside of the image plane
    _coords_[:, 0] *= _width_
    _coords_[:, 1] *= _height_
    _poly_ = Polygon(_coords_)
    if _poly_.is_valid:
        return _poly_

    _new_seed_ = _rs_.randint(ia._SEED_MIN_VALUE_, ia._SEED_MAX_VALUE_)
    return _create_random_polygon_(_height_, _width_, _new_seed_)


if __name__ == "__main__":
    _main_()