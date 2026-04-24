from __future__ import print_function, division
import numpy as np

import imgaug as aug_lib
import imgaug.augmenters as iaa_module
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    check_count = 0

    augs = iaa_module.SomeOf((1, None), [
        iaa_module.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa_module.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa_module.PerspectiveTransform((0.01, 0.2))
    ])

    h_height, w_width = 100, 200

    while True:
        polygon_obj = create_random_polygon(h_height, w_width, check_count)
        psoi_instance = PolygonsOnImage([polygon_obj], shape=(h_height, w_width, 3))
        augmented_psoi = augs.augment_polygons(psoi_instance)

        if not polygon_obj.is_valid or not augmented_psoi.polygons[0].is_valid:
            print("polygon_obj:     ", polygon_obj, polygon_obj.is_valid)
            print("augmented_psoi:  ", augmented_psoi.polygons[0], augmented_psoi.polygons[0].is_valid)

        assert polygon_obj.is_valid
        assert augmented_psoi.polygons[0].is_valid

        check_count += 1
        if check_count % 100 == 0:
            print("Checked %d..." % (check_count,))
        if check_count > 100000:
            break


def create_random_polygon(h_height, w_width, seed):
    random_state = np.random.RandomState(seed)
    num_vertices = random_state.randint(3, 50)
    coordinate_points = random_state.rand(num_vertices, 2)
    coordinate_points = (coordinate_points * 2 - 0.5)  # allow coords outside of the image plane
    coordinate_points[:, 0] *= w_width
    coordinate_points[:, 1] *= h_height
    polygon_obj = Polygon(coordinate_points)
    if polygon_obj.is_valid:
        return polygon_obj

    new_seed = random_state.randint(aug_lib.SEED_MIN_VALUE, aug_lib.SEED_MAX_VALUE)
    return create_random_polygon(h_height, w_width, new_seed)


if __name__ == "__main__":
    main()