from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def main():
    _checkedCountVar = 0

    _someOfAugmenter_v01 = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    _imgHeightDim = 100
    _imgWidthDim = 200

    while True:
        _randomPoly_v01 = create_random_polygon(_imgHeightDim, _imgWidthDim, _checkedCountVar)
        _polygonsOnImageObj_v01 = PolygonsOnImage([_randomPoly_v01], shape=(100, 200, 3))
        _augmentedPolygonsObj_v01 = _someOfAugmenter_v01.augment_polygons(_polygonsOnImageObj_v01)

        if not _randomPoly_v01.is_valid or not _augmentedPolygonsObj_v01.polygons[0].is_valid:
            print("poly:     ", _randomPoly_v01, _randomPoly_v01.is_valid)
            print("poly_aug: ", _augmentedPolygonsObj_v01.polygons[0], _augmentedPolygonsObj_v01.polygons[0].is_valid)

        assert _randomPoly_v01.is_valid
        assert _augmentedPolygonsObj_v01.polygons[0].is_valid

        _checkedCountVar += 1
        if _checkedCountVar % 100 == 0:
            print("Checked %d..." % (_checkedCountVar,))
        if _checkedCountVar > 100000:
            break


def create_random_polygon(_imgHeightDim, _imgWidthDim, _seedParam):
    _randomStateObj_v01 = np.random.RandomState(_seedParam)
    _nbPointsCount_v01 = _randomStateObj_v01.randint(3, 50)
    _coordsArray_v01 = _randomStateObj_v01.rand(_nbPointsCount_v01, 2)
    _coordsArray_v01 = (_coordsArray_v01 * 2 - 0.5)  # allow coords outside of the image plane
    _coordsArray_v01[:, 0] *= _imgWidthDim
    _coordsArray_v01[:, 1] *= _imgHeightDim
    _newPolygon_v01 = Polygon(_coordsArray_v01)
    if _newPolygon_v01.is_valid:
        return _newPolygon_v01

    _newSeedParam = _randomStateObj_v01.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return create_random_polygon(_imgHeightDim, _imgWidthDim, _newSeedParam)


if __name__ == "__main__":
    main()