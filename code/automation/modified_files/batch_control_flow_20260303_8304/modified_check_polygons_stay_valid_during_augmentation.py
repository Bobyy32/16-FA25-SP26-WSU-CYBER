from __future__ import print_function, division
import numpy as np


import imgaug as image_augmentation
import imgaug.augmenters as augmentation_transformer
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    check_count_variable = 0

    augm_obj = augmentation_transformer.SomeOf((1, None), [
        augmentation_transformer.Resize({"height": (1, 100), "width": (1, 100)}),
        augmentation_transformer.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        augmentation_transformer.PerspectiveTransform((0.01, 0.2))
    ])

    vertical_dimens_x, horizontal_px_width = 100, 200

    while True:
        polygon_obj_instance = create_random_polygon(vertical_dimens_x, horizontal_px_width, check_count_variable)
        polygon_struct_on_im = PolygonsOnImage([polygon_obj_instance], shape=(vertical_dimens_x, horizontal_px_width, 3))
        augmented_polygon_obj = augm_obj.augment_polygons(polygon_struct_on_im)

        if not polygon_obj_instance.is_valid or not augmented_polygon_obj.polygons[0].is_valid:
            print("poly:     ", polygon_obj_instance, polygon_obj_instance.is_valid)
            print("poly_aug: ", augmented_polygon_obj.polygons[0], augmented_polygon_obj.polygons[0].is_valid)

        assert polygon_obj_instance.is_valid
        assert augmented_polygon_obj.polygons[0].is_valid

        check_count_variable += 1
        if check_count_variable % 100 == 0:
            print("Checked %d..." % (check_count_variable,))
        if check_count_variable > 100000:
            break


def create_random_polygon(vertical_dimens_x, horizontal_px_width, seed_value):
    random_state_obj = np.random.RandomState(seed_value)
    point_count_range = random_state_obj.randint(3, 50)
    coordinate_array = random_state_obj.rand(point_count_range, 2)
    coordinate_array = (coordinate_array * 2 - 0.5)  # allow coords outside of the image plane
    coordinate_array[:, 0] *= horizontal_px_width
    coordinate_array[:, 1] *= vertical_dimens_x
    polygon_obj_instance = Polygon(coordinate_array)
    if polygon_obj_instance.is_valid:
        return polygon_obj_instance

    new_random_seed = random_state_obj.randint(image_augmentation.SEED_MIN_VALUE, image_augmentation.SEED_MAX_VALUE)
    return create_random_polygon(vertical_dimens_x, horizontal_px_width, new_random_seed)


if __name__ == "__main__":
    main()