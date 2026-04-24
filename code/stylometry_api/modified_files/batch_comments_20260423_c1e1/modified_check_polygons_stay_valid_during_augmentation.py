"""
Main entry point for the polygon augmentation testing utility.
This module demonstrates random polygon generation with image augmentation.
"""

from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    """
    Execute the main processing loop for polygon generation and augmentation testing.
    
    Continuously generates random polygons, applies augmentations, and tracks
    validity checks until reaching a threshold of checks or stopping condition.
    """
    nb_checked = 0

    # Initialize augmentation pipeline with random transformation selections
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

    # Set up the target image dimensions for polygon creation
    height, width = 100, 200

    # Iterate through polygon generation and augmentation verification loop
    while True:
        poly = create_random_polygon(height, width, nb_checked)
        psoi = PolygonsOnImage([poly], shape=(height, width, 3))
        psoi_aug = augs.augment_polygons(psoi)

        # Validate that original and augmented polygon shapes remain valid
        if not poly.is_valid or not psoi_aug.polygons[0].is_valid:
            print("poly:     ", poly, poly.is_valid)
            print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        # Confirm both polygon states maintain their expected validity
        assert poly.is_valid
        assert psoi_aug.polygons[0].is_valid

        nb_checked += 1
        if nb_checked % 100 == 0:
            print("Checked %d..." % (nb_checked,))
        if nb_checked > 100000:
            break


def create_random_polygon(height, width, seed):
    """
    Generate a random polygon within the specified image dimensions.
    
    Creates polygons using random coordinate values that may extend beyond
    the image boundaries to ensure robustness of augmentation processes.
    Recursively attempts generation until a valid polygon configuration is achieved.
    
    Args:
        height: Image height dimension measured in pixels
        width: Image width dimension measured in pixels
        seed: Random state seed for reproducible generation
    
    Returns:
        A Polygon object that passes validation checks
    """
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # enable coordinates positioned outside the image boundaries
    coords[:, 0] *= width
    coords[:, 1] *= height
    poly = Polygon(coords)
    if poly.is_valid:
        return poly

    # Attempt new random state to retry polygon construction
    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    main()