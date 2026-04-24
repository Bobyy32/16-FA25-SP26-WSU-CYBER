from __future__ import print_function, division
import numpy as numpy_lib

import imgaug as ia_module
import imgaug.augmenters as aug_lib
from imgaug.augmentables.polys import Poly_Class, PolyOnIm_Class


def execute_program():
    cnt_checked = 0

    aug_set = aug_lib.SomeOf((1, None), [
        aug_lib.Resize({"height": (1, 100), "width": (1, 100)}),
        aug_lib.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        aug_lib.PerspectiveTransform((0.01, 0.2))
    ])

    hgt, wth = 100, 200

    while True:
        p = generate_random_poly(hgt, wth, cnt_checked)
        psoi_obj = PolyOnIm_Class([p], shape=(hgt, wth, 3))
        psoi_aug_obj = aug_set.augment_polygons(psoi_obj)

        if not p.is_valid or not psoi_aug_obj.polygons[0].is_valid:
            print("poly:     ", p, p.is_valid)
            print("poly_aug: ", psoi_aug_obj.polygons[0], psoi_aug_obj.polygons[0].is_valid)

        assert p.is_valid
        assert psoi_aug_obj.polygons[0].is_valid

        cnt_checked += 1
        if cnt_checked % 100 == 0:
            print("Checked %d..." % (cnt_checked,))
        if cnt_checked > 100000:
            break


def generate_random_poly(hgt, wth, seed):
    rs_obj = numpy_lib.random.RandomState(seed)
    n_pts = rs_obj.randint(3, 50)
    csd_pts = rs_obj.rand(n_pts, 2)
    csd_pts = (csd_pts * 2 - 0.5)  # allow coords outside of the image plane
    csd_pts[:, 0] *= wth
    csd_pts[:, 1] *= hgt
    p = Poly_Class(csd_pts)
    if p.is_valid:
        return p

    new_seed = rs_obj.randint(ia_module.SEED_MIN_VALUE, ia_module.SEED_MAX_VALUE)
    return generate_random_poly(hgt, wth, new_seed)


if __name__ == "__main__":
    execute_program()