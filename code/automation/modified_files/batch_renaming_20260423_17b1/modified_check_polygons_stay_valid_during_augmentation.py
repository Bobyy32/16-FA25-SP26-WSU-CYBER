from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def _execute_core_pipeline():
    _check_counter = 0

    _data_augmenter = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    _img_height, _img_width = 100, 200

    while True:
        _poly_spec = create_random_polygon(_img_height, _img_width, _check_counter)
        _polygon_obj = PolygonsOnImage([_poly_spec], shape=(_img_height, _img_width, 3))
        _polygon_obj_aug = _data_augmenter.augment_polygons(_polygon_obj)

        if not _poly_spec.is_valid or not _polygon_obj_aug.polygons[0].is_valid:
            print("poly:     ", _poly_spec, _poly_spec.is_valid)
            print("poly_aug: ", _polygon_obj_aug.polygons[0], _polygon_obj_aug.polygons[0].is_valid)

        assert _poly_spec.is_valid
        assert _polygon_obj_aug.polygons[0].is_valid

        _check_counter += 1
        if _check_counter % 100 == 0:
            print("Checked %d..." % (_check_counter,))
        if _check_counter > 100000:
            break


def generate_random_shape_spec(_img_height, _img_width, _seed):
    _random_state = np.random.RandomState(_seed)
    _num_points = _random_state.randint(3, 50)
    _point_coords = _random_state.rand(_num_points, 2)
    _point_coords = (_point_coords * 2 - 0.5)  # allow coords outside of the image plane
    _point_coords[:, 0] *= _img_width
    _point_coords[:, 1] *= _img_height
    _poly_spec = Polygon(_point_coords)
    if _poly_spec.is_valid:
        return _poly_spec

    _new_seed = _random_state.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return generate_random_shape_spec(_img_height, _img_width, _new_seed)


if __name__ == "__main__":
    _execute_core_pipeline()