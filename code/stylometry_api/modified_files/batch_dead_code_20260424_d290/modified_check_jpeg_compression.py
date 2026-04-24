from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


class ValueWrapper(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return repr(self.value)


class ConfigWrapper(object):
    def __init__(self, config):
        self.config = config


def main():
    augs = [
        ("0", ValueWrapper(iaa.JpegCompression(compression=0))),
        ("1", ValueWrapper(iaa.JpegCompression(compression=1))),
        ("25", ValueWrapper(iaa.JpegCompression(compression=25))),
        ("50", ValueWrapper(iaa.JpegCompression(compression=50))),
        ("75", ValueWrapper(iaa.JpegCompression(compression=75))),
        ("99", ValueWrapper(iaa.JpegCompression(compression=99))),
        ("100", ValueWrapper(iaa.JpegCompression(compression=100))),
        ("(0, 50)", ValueWrapper(iaa.JpegCompression(compression=(0, 50)))),
        ("(50, 100)", ValueWrapper(iaa.JpegCompression(compression=(50, 100)))),
        ("(0, 100)", ValueWrapper(iaa.JpegCompression(compression=(0, 100)))),
    ]

    image = ia.data.quokka(size=(256, 256), extract="square")
    images = np.uint8([image] * (5*5))

    for i, (name, aug) in enumerate(augs):
        print(i, name)
        images_aug = aug.augment_images(images)
        ia.imshow(ia.draw_grid(images_aug, cols=5, rows=5))


if __name__ == "__main__":
    main()