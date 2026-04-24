from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


"""
This script generates and validates random polygons, testing augmentation effects.
It creates random polygon shapes within a specified image grid, applies various
augmentation transformations, and verifies that valid polygons remain valid after
processing. The script continues checking new instances until reaching a target
limit or encountering validation issues.
"""


def main():
    """
    Main execution function for generating and validating augmented polygons.
    Iterates through random polygon creations, augments them using various
    image augmentation layers, and checks for validity preservation. Tracks
    the number of successful validations and breaks after exceeding the
    defined threshold limit.
    """

    nb_checked = 0

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

    height, width = 100, 200

    while True:
        poly = create_random_polygon(height, width, nb_checked)
        psoi = PolygonsOnImage([poly], shape=(height, width, 3))
        psoi_aug = augs.augment_polygons(psoi)

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


def create_random_polygon(height, width, seed):
    """
    Generates a single random polygon within the specified dimensions.
    Uses a seeded random state to ensure reproducibility when needed.
    Creates polygon coordinates by sampling points across the entire image
    area, allowing for boundary extrapolation to simulate realistic conditions.
    Repeats the generation process with a new random seed if invalid results
    occur, ensuring a valid polygon is returned.
    """
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coordinates beyond the image bounds
    coords[:, 0] *= width
    coords[:, 1] *= height
    poly = Polygon(coords)
    if poly.is_valid:
        return poly

    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    main()