from __future__ import print_function, division
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

def processImage():
    totalProcessed = 0
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
        currentPolygon = generateRandomPolygon(imgHeight, imgWidth, totalProcessed)
        polygonsOnImage = PolygonsOnImage([currentPolygon], shape=(imgHeight, imgWidth, 3))
        augmentedPolygons = augmentations.augment_polygons(polygonsOnImage)
        if not currentPolygon.is_valid or not augmentedPolygons.polygons[0].is_valid:
            print("poly:     ", currentPolygon, currentPolygon.is_valid)
            print("poly_aug: ", augmentedPolygons.polygons[0], augmentedPolygons.polygons[0].is_valid)
        assert currentPolygon.is_valid
        assert augmentedPolygons.polygons[0].is_valid
        totalProcessed += 1
        if totalProcessed % 100 == 0:
            print("Checked %d..." % (totalProcessed,))
        if totalProcessed > 100000:
            break

def generateRandomPolygon(imgHeight, imgWidth, seedValue):
    randomState = np.random.RandomState(seedValue)
    pointCount = randomState.randint(3, 50)
    coordinates = randomState.rand(pointCount, 2)
    coordinates = (coordinates * 2 - 0.5)
    coordinates[:, 0] *= imgWidth
    coordinates[:, 1] *= imgHeight
    polygon = Polygon(coordinates)
    if polygon.is_valid:
        return polygon
    newSeed = randomState.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return generateRandomPolygon(imgHeight, imgWidth, newSeed)

if __name__ == "__main__":
    processImage()