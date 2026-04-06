from __future__ import print_function, division
import numpy as np

from imgaug.augmenters import ia
from imgaug.augmentables.polys import PolygonsOnImage, Polygon

def main():
    nb_checked = 0

    augs_aug = ia.SomeOf((1, None), [
        ia.Resize({"height": (1, 100), "width": (1, 100)}),
        ia.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        ia.PerspectiveTransform((0.01, 0.2))
    ])

    height_aug, width_aug = 100, 200

    while True:
        poly = create_random_polygon(height_aug, width_aug, nb_checked)
        psoi_aug = PolygonsOnImage([poly], shape=(height_aug, width_aug, 3))
        psoi_aug = augs_aug.augment_polygons(psoi_aug)

        if not poly.is_valid or not psoi_aug.polygons[0].is_valid:
            print("poly:     ", poly, poly.is_valid)
            print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        assert poly.is_valid
        assert psoi_aug.polygons[0].is_valid

        nb_checked += 1
        if nb_checked % 100 == 0:
            print("Checked %d..." % (nb_checked,))
        if nb_checked > 100000:
            break


def create_random_polygon(height_aug, width_aug, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 1] *= height_aug
    coords[:, 0] *= width_aug
    poly = Polygon(coords)
    if poly.is_valid:
        return poly

    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(height_aug, width_aug, new_seed)


if __name__ == "__main__":
    main()