from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def executionEntryPoint():
    counter = 0

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

    imgHeight, imgWidth = 100, 200

    while True:
        polygon = generateRandomPolygon(imgHeight, imgWidth, counter)
        polygonsOnImage = PolygonsOnImage([polygon], shape=(imgHeight, imgWidth, 3))
        augmentedPolygons = augmentations.augment_polygons(polygonsOnImage)

        if not polygon.is_valid or not augmentedPolygons.polygons[0].is_valid:
            print("polygon:     ", polygon, polygon.is_valid)
            print("polygon_aug: ", augmentedPolygons.polygons[0], augmentedPolygons.polygons[0].is_valid)

        assert polygon.is_valid
        assert augmentedPolygons.polygons[0].is_valid

        counter += 1
        if counter % 100 == 0:
            print("Checked %d..." % (counter,))
        if counter > 100000:
            break


def generateRandomPolygon(height, width, seed):
    randomState = np.random.RandomState(seed)
    pointCount = randomState.randint(3, 50)
    coordinates = randomState.rand(pointCount, 2)
    coordinates = (coordinates * 2 - 0.5)  # allow coordinates outside of the image plane
    coordinates[:, 0] *= width
    coordinates[:, 1] *= height
    polygon = Polygon(coordinates)
    if polygon.is_valid:
        return polygon

    newSeed = randomState.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return generateRandomPolygon(height, width, newSeed)


if __name__ == "__main__":
    executionEntryPoint()