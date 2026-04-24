from __future__ import print_function, division
import numpy as np

import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.polys import PolygonsOnImage, Polygon


def main():
    checked_count = 0

    augs = iaa.SomeOf((1, None), [
        iaa.Affine(
            translate_px={"x": (-50, 50), "y": (-50, 50)},
            scale=(0.01, 2.0),
            shear=(-360, 360),
            rotate=(-360, 360)
        ),
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    height_val, width_val = 100, 200

    while True:
        polygon = create_random_polygon(height_val, width_val, checked_count)
        psoi_obj = PolygonsOnImage([polygon], shape=(height_val, width_val, 3))
        augmented_polygons = augs.augment_polygons(psoi_obj)

        if not polygon.is_valid or not augmented_polygons.polygons[0].is_valid:
            print("poly:     ", polygon, polygon.is_valid)
            print("augmented: ", augmented_polygons.polygons[0], augmented_polygons.polygons[0].is_valid)

        assert polygon.is_valid
        assert augmented_polygons.polygons[0].is_valid

        checked_count += 1
        if checked_count % 100 == 0:
            print("Checked %d..." % (checked_count,))
        if checked_count > 100000:
            break


def create_random_polygon(height, width, seed):
    """Generates polygon with random coordinates within bounds."""
    rs = np.random.RandomState(seed)
    point_count = rs.randint(3, 50)
    coords = rs.rand(point_count, 2)
    coords = (coords * 2 - 0.5)  # permit coordinates outside the image plane
    coords[:, 1] *= height
    coords[:, 0] *= width
    poly_instance = Polygon(coords)
    if poly_instance.is_valid:
        return poly_instance

    replacement_seed = rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(height, width, replacement_seed)


if __name__ == "__main__":
    main()