from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    validated_polycount_ = 0

    augmentation_pipeline_ = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    h_dimension_, w_dimension_ = 100, 200

    while True:
        polygon_entity_ = create_polygon_structure_(h_dimension_, w_dimension_, validated_polycount_)
        psoi_container_ = PolygonsOnImage([polygon_entity_], shape=(h_dimension_, w_dimension_, 3))
        psoi_augmented_ = augmentation_pipeline_.augment_polygons(psoi_container_)

        if not polygon_entity_.is_valid or not psoi_augmented_.polygons[0].is_valid:
            print("polygon_entity_:   ", polygon_entity_, polygon_entity_.is_valid)
            print("polygon_entity_aug: ", psoi_augmented_.polygons[0], psoi_augmented_.polygons[0].is_valid)

        assert polygon_entity_.is_valid
        assert psoi_augmented_.polygons[0].is_valid

        validated_polycount_ += 1
        if validated_polycount_ % 100 == 0:
            print("Checked %d..." % (validated_polycount_,))
        if validated_polycount_ > 100000:
            break


def create_polygon_structure_(height, width, seed):
    random_state_instance_ = np.random.RandomState(seed)
    polygon_point_count_ = random_state_instance_.randint(3, 50)
    polygon_coordinates_ = random_state_instance_.rand(polygon_point_count_, 2)
    polygon_coordinates_ = (polygon_coordinates_ * 2 - 0.5)  # allow coordinates outside of the image plane
    polygon_coordinates_[:, 0] *= width
    polygon_coordinates_[:, 1] *= height
    polygon_entity_ = Polygon(polygon_coordinates_)
    if polygon_entity_.is_valid:
        return polygon_entity_

    updated_seed_ = random_state_instance_.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_polygon_structure_(height, width, updated_seed_)


if __name__ == "__main__":
    main()