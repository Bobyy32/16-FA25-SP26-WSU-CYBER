from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    inspection_counter = 0

    augmenter_collection = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    vertical_size, horizontal_size = 100, 200

    while True:
        random_polygon_object = create_random_polygon_primitive(vertical_size, horizontal_size, inspection_counter)
        polygon_set_on_image_obj = PolygonsOnImage([random_polygon_object], shape=(vertical_size, horizontal_size, 3))
        augmented_polygons_collection = augmenter_collection.augment_polygons(polygon_set_on_image_obj)

        if not random_polygon_object.is_valid or not augmented_polygons_collection.polygons[0].is_valid:
            print("polygon_primitive:", random_polygon_object, random_polygon_object.is_valid)
            print("augmented_primitive:", augmented_polygons_collection.polygons[0], augmented_polygons_collection.polygons[0].is_valid)

        assert random_polygon_object.is_valid and augmented_polygons_collection.polygons[0].is_valid

        inspection_counter += 1
        if inspection_counter % 100 == 0:
            print("Evaluated {} items so far...".format(inspection_counter))
        if inspection_counter > 100000:
            break


def create_random_polygon_primitive(vertical_size, horizontal_size, iteration_seed):
    random_state = np.random.RandomState(iteration_seed)
    vertex_count = random_state.randint(3, 50)
    vertex_coordinates = random_state.rand(vertex_count, 2)
    vertex_coordinates = (vertex_coordinates * 2 - 0.5)  # allow coords outside of the image plane
    vertex_coordinates[:, 0] *= horizontal_size
    vertex_coordinates[:, 1] *= vertical_size
    polygon_primitive = Polygon(vertex_coordinates)
    if polygon_primitive.is_valid:
        return polygon_primitive

    new_seed_value = random_state.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon_primitive(vertical_size, horizontal_size, new_seed_value)


if __name__ == "__main__":
    main()