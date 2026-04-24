# This Python script generates and validates random polygons with image augmentations

from __future__ import print_function, division
import numpy as np

# Additional libraries for image augmentations and polygon handling
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    """
    Main function to test polygon augmentation with random generated polygons
    Iterates through polygon validation to ensure augmentations preserve validity
    """
    nb_checked = 0

    # Define augmentation pipeline with various transforms
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

    # Set image dimensions for polygon generation
    height, width = 100, 200

    # Generate and validate polygons in an infinite loop
    while True:
        poly = create_random_polygon(height, width, nb_checked)
        psoi = PolygonsOnImage([poly], shape=(height, width, 3))
        psoi_aug = augs.augment_polygons(psoi)

        # Check if polygon or augmented polygon is invalid
        if not poly.is_valid or not psoi_aug.polygons[0].is_valid:
            # Log the validation details for debugging
            print("poly:     ", poly, poly.is_valid)
            print("poly_aug: ", psoi_aug.polygons[0], psoi_aug.polygons[0].is_valid)

        # Ensure both original and augmented polygons remain valid
        assert poly.is_valid
        assert psoi_aug.polygons[0].is_valid

        nb_checked += 1
        # Progress report every 100 iterations
        if nb_checked % 100 == 0:
            print("Checked %d..." % (nb_checked,))
        # Stop after checking 100k polygons
        if nb_checked > 100000:
            break


def create_random_polygon(height, width, seed):
    """
    Creates a random polygon within specified image dimensions
    Recursively tries new seed if polygon generation fails validation
    """
    rs = np.random.RandomState(seed)
    nb_points = rs.randint(3, 50)
    coords = rs.rand(nb_points, 2)
    coords = (coords * 2 - 0.5)  # allow coords outside of the image plane
    coords[:, 0] *= width
    coords[:, 1] *= height
    poly = Polygon(coords)
    if poly.is_valid:
        return poly

    # Generate new seed if polygon is invalid
    new_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    main()