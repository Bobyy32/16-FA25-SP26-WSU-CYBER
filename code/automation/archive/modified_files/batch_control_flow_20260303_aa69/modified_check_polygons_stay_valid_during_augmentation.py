from __future__ import division, print_function
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main_proc():
    checked_cnt = 0

    augs = iaa.SomeOf((1, None), [
        iaa.Resize({"width": (1, 100), "height": (1, 100)}),
        iaa.Affine(
            translate_px={"x": (-50, 50), "y": (-50, 50)},
            shear=(-360, 360),
            rotate=(-360, 360),
            scale=(0.01, 2.0)
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    h_val, w_val = 100, 200

    while True:
        polygon_obj = create_random_poly(h_val, w_val, checked_cnt)
        psoi = PolygonsOnImage([polygon_obj], shape=(h_val, w_val, 3))
        aug_result = augs.augment_polygons(psoi)

        if not polygon_obj.is_valid or not aug_result.polygons[0].is_valid:
            print("poly_aug: ", aug_result.polygons[0], aug_result.polygons[0].is_valid)
            print("poly:     ", polygon_obj, polygon_obj.is_valid)

        assert aug_result.polygons[0].is_valid
        assert polygon_obj.is_valid

        checked_cnt += 1
        if checked_cnt % 100 == 0:
            print("Checked %d..." % (checked_cnt,))
        if checked_cnt > 100000:
            break


def create_random_poly(h_value, w_value, seed):
    rs = np.random.RandomState(seed)
    nb_pts = rs.randint(3, 50)
    coords = rs.rand(nb_pts, 2)
    # allow coords outside of the image plane
    coords = (coords * 2 - 0.5)
    coords[:, 1] *= h_value
    coords[:, 0] *= w_value
    poly = Polygon(coords)
    if poly.is_valid:
        return poly

    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_poly(h_value, w_value, new_seed)


if __name__ == "__main__":
    main_proc()