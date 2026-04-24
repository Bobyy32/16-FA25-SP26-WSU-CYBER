from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def execute_main():
    count_validated = 0

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

    h_dim, w_dim = 100, 200

    while True:
        poly_shape = create_random_poly_shape(h_dim, w_dim, count_validated)
        psoi_aug = PolygonsOnImage([poly_shape], shape=(h_dim, w_dim, 3))
        psoi_augs = augs.augment_polygons(psoi_aug)

        if not poly_shape.is_valid or not psoi_augs.polygons[0].is_valid:
            print("poly_shape:     ", poly_shape, poly_shape.is_valid)
            print("poly_aug:       ", psoi_augs.polygons[0], psoi_augs.polygons[0].is_valid)

        assert poly_shape.is_valid
        assert psoi_augs.polygons[0].is_valid

        count_validated += 1
        if count_validated % 100 == 0:
            print("Checked %d..." % (count_validated,))
        if count_validated > 100000:
            break


def create_random_poly_shape(h_dim, w_dim, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 0] *= w_dim
    coords[:, 1] *= h_dim
    poly = Polygon(coords)
    if poly.is_valid:
        return poly

    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_poly_shape(h_dim, w_dim, new_seed)


if __name__ == "__main__":
    execute_main()