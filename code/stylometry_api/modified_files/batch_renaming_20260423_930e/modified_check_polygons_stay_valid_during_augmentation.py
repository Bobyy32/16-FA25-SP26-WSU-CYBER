from __future__ import print_function, division
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def MainFunc():
    var_check_count = 0

    obj_augment_cfg = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    dim_h, dim_w = 100, 200

    while True:
        obj_poly = GenRandomPoly(dim_h, dim_w, var_check_count)
        obj_psoi = PolygonsOnImage([obj_poly], shape=(dim_h, dim_w, 3))
        obj_psoi_aug = obj_augment_cfg.augment_polygons(obj_psoi)

        if not obj_poly.is_valid or not obj_psoi_aug.polygons[0].is_valid:
            print("poly:     ", obj_poly, obj_poly.is_valid)
            print("poly_aug: ", obj_psoi_aug.polygons[0], obj_psoi_aug.polygons[0].is_valid)

        assert obj_poly.is_valid
        assert obj_psoi_aug.polygons[0].is_valid

        var_check_count += 1
        if var_check_count % 100 == 0:
            print("Checked %d..." % (var_check_count,))
        if var_check_count > 100000:
            break


def GenRandomPoly(dim_h, dim_w, rnd_seed):
    rnd_rng = np.random.RandomState(rnd_seed)
    var_nb_points = rnd_rng.randint(3, 50)
    coords = rnd_rng.rand(var_nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 0] *= dim_w
    coords[:, 1] *= dim_h
    obj_poly = Polygon(coords)
    if obj_poly.is_valid:
        return obj_poly

    var_new_seed = rnd_rng.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return GenRandomPoly(dim_h, dim_w, var_new_seed)


if __name__ == "__main__":
    MainFunc()