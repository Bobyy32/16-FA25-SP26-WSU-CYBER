from __future__ import print_function, division
import XNumPy

import XImgAug
import XImgAugAug
from XImgAugAug.polys import XPolygon, XPolygonsOnImage


def XMain():
    X_nb_checked = 0

    X_augs = XImgAugSomeOf((1, None), [
        XImgAugResize({"X_height": (1, 100), "X_width": (1, 100)}),
        XImgAugAffine(
            X_scale=(0.01, 2.0),
            X_rotate=(-360, 360),
            X_shear=(-360, 360),
            X_translate_px={"X_x": (-50, 50), "X_y": (-50, 50)}
        ),
        XImgAugPerspectiveTransform((0.01, 0.2))
    ])

    X_height, X_width = 100, 200

    while True:
        X_poly = XCreateRandomPolygon(X_height, X_width, X_nb_checked)
        X_psoi = XPolygonsOnImage([X_poly], shape=(X_height, X_width, 3))
        X_psoi_aug = X_augs.X_augment_polygons(X_psoi)

        if not X_poly.X_is_valid or not X_psoi_aug.X_polygons[0].X_is_valid:
            print("X_poly:     ", X_poly, X_poly.X_is_valid)
            print("X_poly_aug: ", X_psoi_aug.X_polygons[0], X_psoi_aug.X_polygons[0].X_is_valid)

        assert X_poly.X_is_valid
        assert X_psoi_aug.X_polygons[0].X_is_valid

        X_nb_checked += 1
        if X_nb_checked % 100 == 0:
            print("Checked %d..." % (X_nb_checked,))
        if X_nb_checked > 100000:
            break


def XCreateRandomPolygon(X_height, X_width, X_seed):
    X_rs = XNumPy.random.RandomState(X_seed)
    X_nb_points = X_rs.randint(3, 50)
    X_coords = X_rs.rand(X_nb_points, 2)
    X_coords = (X_coords * 2 - 0.5)  # allow X_coords outside of the X_image plane
    X_coords[:, 0] *= X_width
    X_coords[:, 1] *= X_height
    X_poly = XPolygon(X_coords)
    if X_poly.X_is_valid:
        return X_poly

    X_new_seed = X_rs.randint(XImgAugSomeOfSEED_MIN_VALUE, XImgAugSomeOfSEED_MAX_VALUE)
    return XCreateRandomPolygon(X_height, X_width, X_new_seed)


if __name__ == "__XMain":
    XMain()