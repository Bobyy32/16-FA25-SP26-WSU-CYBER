from __future__ import print_function, division
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    total_count = 0
    augmentations = iaa.SomeOf((1, None), [
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
    while total_count < 100000:
        current_polygon = create_random_polygon(img_height, img_width, total_count)
        polygons_image = PolygonsOnImage([current_polygon], shape=(img_height, img_width, 3))
        augmented_polygons = augmentations.augment_polygons(polygons_image)
        if not current_polygon.is_valid or not augmented_polygons.polygons[0].is_valid:
            print("poly:     ", current_polygon, current_polygon.is_valid)
            print("poly_aug: ", augmented_polygons.polygons[0], augmented_polygons.polygons[0].is_valid)
        assert current_polygon.is_valid
        assert augmented_polygons.polygons[0].is_valid
        total_count += 1
        if total_count % 100 == 0:
            print("Checked %d..." % (total_count,))


def create_random_polygon(height, width, seed_value):
    random_state = np.random.RandomState(seed_value)
    point_count = random_state.randint(3, 50)
    coordinates = random_state.rand(point_count, 2)
    coordinates = (coordinates * 2 - 0.5)
    coordinates[:, 0] *= width
    coordinates[:, 1] *= height
    polygon = Polygon(coordinates)
    if polygon.is_valid:
        return polygon
    new_seed = random_state.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(height, width, new_seed)


if __name__ == "__main__":
    main()