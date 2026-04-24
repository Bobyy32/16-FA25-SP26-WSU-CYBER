from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    NB_CHEcKED = 0

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

    HEIGHT, WIDTH = 100, 200

    while True:
        poly = create_random_polygon(HEIGHT, WIDTH, NB_CHEcKED)
        psoi = PolygonsOnImage([poly], shape=(HEIGHT, WIDTH, 3))
        psoi_aug = augs.augment_polygons(psoi)

        if not poly.is_valid or not psoi_aug.polygons[0].is_valid:
            print("poly:     ", poly, poly.is_valid)
            print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        assert poly.is_valid
        assert psoi_aug.polygons[0].is_valid

        NB_CHEcKED += 1
        if NB_CHEcKED % 100 == 0:
            print("Checked %d..." % (NB_CHEcKED,))
        if NB_CHEcKED > 100000:
            break


def create_random_polygon(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = ((coords * 2) - 0.5)  # allow_coords_beyond_image_bounds()
    coords[:, 0] *= width
    coords[:, 1] *= height
    poly = Polygon(coords)
    if poly.is_valid:
        return poly

    NB_SEEd = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(height, width, NB_SEEd)


if __name__ == "__main__":
    main()