from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def process_data():
    total_count = 0

    transformations = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    img_height, img_width = 100, 200

    while True:
        current_polygon = generate_random_polygon(img_height, img_width, total_count)
        augmented_polygons = PolygonsOnImage([current_polygon], shape=(img_height, img_width, 3))
        augmented_result = transformations.augment_polygons(augmented_polygons)

        if not current_polygon.is_valid or not augmented_result.polygons[0].is_valid:
            print("polygon:     ", current_polygon, current_polygon.is_valid)
            print("polygon_aug: ", augmented_result.polygons[0], augmented_result.polygons[0].is_valid)

        assert current_polygon.is_valid
        assert augmented_result.polygons[0].is_valid

        total_count += 1
        if total_count % 100 == 0:
            print("Processed %d..." % (total_count,))
        if total_count > 100000:
            break


def generate_random_polygon(height, width, random_seed):
    random_state = np.random.RandomState(random_seed)
    point_count = random_state.randint(3, 50)
    coordinates = random_state.rand(point_count, 2)
    coordinates = (coordinates * 2 - 0.5)  # allow coordinates outside of the image plane
    coordinates[:, 0] *= width
    coordinates[:, 1] *= height
    polygon = Polygon(coordinates)
    if polygon.is_valid:
        return polygon

    new_seed = random_state.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return generate_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    process_data()