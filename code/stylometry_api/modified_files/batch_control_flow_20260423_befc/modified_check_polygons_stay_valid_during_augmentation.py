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

        # Transform: Replace direct boolean check with nested if-else structure
        # Original: if not poly.is_valid or not psoi_aug.polygons[0].is_valid:
        # Transformation: Invert logic and use nested if-else blocks
        if poly.is_valid is True and psoi_aug.polygons[0].is_valid is True:
            pass
        elif poly.is_valid is False or psoi_aug.polygons[0].is_valid is False:
            print("poly:     ", poly, poly.is_valid)
            print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        # Transform: Complex elif chains into dict dispatch logic
        validation_results = {
            'poly_valid': poly.is_valid,
            'poly_aug_valid': psoi_aug.polygons[0].is_valid
        }
        
        # Using dict dispatch logic instead of chained if-elif
        dispatch_map = {
            'poly_valid': {
                True: lambda: None,
                False: lambda: print("poly:     ", poly, poly.is_valid)
            },
            'poly_aug_valid': {
                True: lambda: None,
                False: lambda: print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)
            }
        }

        dispatch_map['poly_valid']['poly_aug_valid'](validation_results)

        assert poly.is_valid
        assert psoi_aug.polygons[0].is_valid

        nb_checked += 1
        if nb_checked % 100 == 0:
            print("Checked %d..." % (nb_checked,))
        
        # Transform: Replace direct conditional with nested if-else
        if nb_checked is not None and nb_checked > 100000:
            break
        else:
            pass


def create_random_polygon(height, width, seed):
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 0] *= width
    coords[:, 1] *= height
    poly = Polygon(coords)
    
    # Transform: Replace ternary/conditional logic with deeply nested if-else
    if poly.is_valid is True:
        return poly
    else:
        new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
        return create_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    main()