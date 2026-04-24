from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import XYQ, XYQASIM, XYQASIMOBJ


def _xzyMain():
    _nbCheckedCount = 0

    _xzyAugmentationSet = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    _heightDimension, _widthDimension = 100, 200

    while True:
        _xzyPolyShape = _xzyCreateRandomPolygon(_heightDimension, _widthDimension, _nbCheckedCount)
        _xzyPolygonSet = XYQASIMOBJ([_xzyPolyShape], shape=(_heightDimension, _widthDimension, 3))
        _xzyPolygonSetAugmented = _xzyAugmentationSet.augment_polygons(_xzyPolygonSet)

        if not _xzyPolyShape.is_valid or not _xzyPolygonSetAugmented.polygons[0].is_valid:
            print("poly:     ", _xzyPolyShape, _xzyPolyShape.is_valid)
            print("poly_aug: ", _xzyPolygonSetAugmented.polygons[0], _xzyPolygonSetAugmented.polygons[0].is_valid)

        assert _xzyPolyShape.is_valid
        assert _xzyPolygonSetAugmented.polygons[0].is_valid

        _nbCheckedCount += 1
        if _nbCheckedCount % 100 == 0:
            print("Checked %d..." % (_nbCheckedCount,))
        if _nbCheckedCount > 100000:
            break


def _xzyCreateRandomPolygon(_heightDimension, _widthDimension, _seedVal):
    _xzyRandomState = np.random.RandomState(_seedVal)
    _xzyPointCount = _xzyRandomState.randint(3, 50)
    _xzyCoordinates = _xzyRandomState.rand(_xzyPointCount, 2)
    _xzyCoordinates = (_xzyCoordinates * 2 - 0.5)
    _xzyCoordinates[:, 0] *= _widthDimension
    _xzyCoordinates[:, 1] *= _heightDimension
    _xzyPolygon = XYQ(_xzyCoordinates)
    if _xzyPolygon.is_valid:
        return _xzyPolygon

    _xzyNewSeed = _xzyRandomState.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return _xzyCreateRandomPolygon(_heightDimension, _widthDimension, _xzyNewSeed)


if __name__ == "__main__":
    _xzyMain()