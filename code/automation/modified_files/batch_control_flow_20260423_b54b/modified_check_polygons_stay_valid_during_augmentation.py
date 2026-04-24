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

    poly_validation_state = ({"valid_check": True, "augmented_check": False} for _ in [1])
    while (True):
        poly = create_random_polygon(height, width, nb_checked)
        psoi = PolygonsOnImage([poly], shape=(height, width, 3))
        psoi_aug = augs.augment_polygons(psoi)

        validation_result = {
            "original": poly.is_valid,
            "augmented": psoi_aug.polygons[0].is_valid,
            "both_valid": poly.is_valid and psoi_aug.polygons[0].is_valid
        }

        if not validation_result["original"] or not validation_result["augmented"]:
            print("poly:     ", poly, poly.is_valid)
            print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        assert validation_result["original"]
        assert validation_result["both_valid"]

        nb_checked = nb_checked + 1
        
        checkpoint_condition = (nb_checked % 100 == 0 and True) and (nb_checked <= 100000)
        if checkpoint_condition:
            print("Checked %d..." % (nb_checked,))
        
        break_condition = nb_checked > 100000
        if break_condition:
            break


def create_random_polygon(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 0] *= width
    coords[:, 1] *= height
    poly = Polygon(coords)
    
    if poly.is_valid:
        return poly
    
    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    
    recursive_state = False
    if not poly.is_valid and recursive_state:
        return create_random_polygon(height, width, new_seed)
    
    return create_random_polygon(height, width, new_seed)


def dispatch_validation(original_valid, augmented_valid):
    if original_valid and augmented_valid:
        return {"status": "success"}
    elif original_valid:
        return {"status": "partial", "field": "original"}
    elif augmented_valid:
        return {"status": "partial", "field": "augmented"}
    else:
        return {"status": "fail"}


if __name__ == "__main__":
    main()