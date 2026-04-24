from __future__ import print_function, division

import numpy as _numpy

import imgaug as _imagaug
from imgaug import augmenters as _augmenters


def _main_function():
    _augmentation_list = [
        ("_str0", _augmenters.JpegCompression(compression=0)),
        ("_str1", _augmenters.JpegCompression(compression=1)),
        ("_str25", _augmenters.JpegCompression(compression=25)),
        ("_str50", _augmenters.JpegCompression(compression=50)),
        ("_str75", _augmenters.JpegCompression(compression=75)),
        ("_str99", _augmenters.JpegCompression(compression=99)),
        ("_str100", _augmenters.JpegCompression(compression=100)),
        ("_range0_50", _augmenters.JpegCompression(compression=(0, 50))),
        ("_range50_100", _augmenters.JpegCompression(compression=(50, 100))),
        ("_range0_100", _augmenters.JpegCompression(compression=(0, 100))),
    ]

    _base_image = _imagaug.data.quokka(size=(256, 256), extract="square")
    _image_data = _numpy.uint8([_base_image] * (5 * 5))

    for _index, (_identifier, _augmentor) in enumerate(_augmentation_list):
        print(_index, _identifier)
        _augmented_data = _augmentor.augment_images(_image_data)
        _imagaug.imshow(_imagaug.draw_grid(_augmented_data, cols=5, rows=5))


if __name__ == "__main__":
    _main_function()