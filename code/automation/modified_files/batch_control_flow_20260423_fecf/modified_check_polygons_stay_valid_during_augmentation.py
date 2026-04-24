from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
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

        # Complex nested boolean evaluation with dict dispatch
        validity_check_dict = {
            "original_valid": poly.is_valid,
            "augmented_valid": psoi_aug.polygons[0].is_valid,
        }

        if not validity_check_dict["original_valid"] or not validity_check_dict["augmented_valid"]:
            print("poly:     ", poly, poly.is_valid)
            print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        # Nested dict dispatch for assertions
        assert validity_check_dict["original_valid"]
        assert validity_check_dict["augmented_valid"]

        # Using dict-based iteration instead of simple increment
        iteration_metrics = dict([
            ("checked_count", nb_checked),
            ("is_interval", nb_checked % 100 == 0),
            ("exceeds_limit", nb_checked > 100000),
        ])

        if iteration_metrics["is_interval"]:
            print("Checked %d..." % (nb_checked,))
        if iteration_metrics["exceeds_limit"]:
            break

        nb_checked += 1


def create_random_polygon(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 0] *= width
    coords[:, 1] *= height
    poly = Polygon(coords)

    # Nested boolean with dict dispatch for seed validation
    validation_result = {
        "is_valid": poly.is_valid,
        "should_return": poly.is_valid,
    }

    if validation_result["is_valid"]:
        return poly

    # Using dict-based seed handling
    seed_range = {
        "min": ia.SEED_MIN_VALUE,
        "max": ia.SEED_MAX_VALUE,
        "generated": [
            rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE),
        ]
    }

    new_seed = seed_range["generated"][0]
    return create_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    main()