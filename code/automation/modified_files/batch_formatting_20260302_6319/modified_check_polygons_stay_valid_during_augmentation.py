from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def process_training_data():
    total_evaluations = 0

    augmentation_pipeline = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    image_height, image_width = 100, 200

    while True:
        test_polygon = generate_random_polygon(image_height, image_width, total_evaluations)
        polygon_container = PolygonsOnImage([test_polygon], shape=(image_height, image_width, 3))
        augmented_container = augmentation_pipeline.augment_polygons(polygon_container)

        if not test_polygon.is_valid or not augmented_container.polygons[0].is_valid:
            print("polygon:     ", test_polygon, test_polygon.is_valid)
            print("polygon_aug: ", augmented_container.polygons[0], augmented_container.polygons[0].is_valid)

        assert test_polygon.is_valid
        assert augmented_container.polygons[0].is_valid

        total_evaluations += 1
        if total_evaluations % 100 == 0:
            print("Evaluated %d..." % (total_evaluations,))
        if total_evaluations > 100000:
            break


def generate_random_polygon(height, width, random_seed):
    random_generator = np.random.RandomState(random_seed)
    point_count = random_generator.randint(3, 50)
    coordinates = random_generator.rand(point_count, 2)
    coordinates = (coordinates * 2 - 0.5)  # allow coordinates outside of the image plane
    coordinates[:, 0] *= width
    coordinates[:, 1] *= height
    test_polygon = Polygon(coordinates)
    if test_polygon.is_valid:
        return test_polygon

    new_seed = random_generator.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return generate_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    process_training_data()