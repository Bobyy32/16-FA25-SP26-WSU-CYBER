from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import PolygonsOnImage, Polygon


def main():
    checkCount = 0

    AugPipeline = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    dimX, dimY = 100, 200

    while True:
        polygonShape = createRandomPolygon(dimX, dimY, checkCount)
        polySetObj = PolygonsOnImage([polygonShape], shape=(dimY, dimX, 3))
        augmentedPolygons = AugPipeline.augmentPolygons(polySetObj)

        if not polygonShape.is_valid or not augmentedPolygons.polygons[0].is_valid:
            print("polygonShape:     ", polygonShape, polygonShape.is_valid)
            print("augmentedPolygons:", augmentedPolygons.polygons[0], augmentedPolygons.polygons[0].is_valid)

        assert polygonShape.is_valid
        assert augmentedPolygons.polygons[0].is_valid

        checkCount += 1
        if checkCount % 100 == 0:
            print("Checked %d..." % (checkCount,))
        if checkCount > 100000:
            break


def createRandomPolygon(dimX, dimY, seed):
    rngInstance = np.random.RandomState(seed)
    pointCount = rngInstance.randint(3, 50)
    coordinateMatrix = rngInstance.rand(pointCount, 2)
    coordinateMatrix = (coordinateMatrix * 2 - 0.5)  # allow coords outside of the image plane
    coordinateMatrix[:, 0] *= dimX
    coordinateMatrix[:, 1] *= dimY
    polygonShape = Polygon(coordinateMatrix)
    if polygonShape.is_valid:
        return polygonShape

    newSeed = rngInstance.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return createRandomPolygon(dimX, dimY, newSeed)


if __name__ == "__main__":
    main()