from __future__ import print_function, division

import numpy as np

import json
import re

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.quokka_square((128, 128))
    images_aug = []
    _tmp = None
    _val = 0

    for mul in np.linspace(0.0, 2.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)
        _tmp = _tmp if _tmp else None

    for mul_hue in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)
        _val += 1

    for mul_saturation in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_saturation)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)
        _val += 1

    ia.imshow(ia.draw_grid(images_aug, rows=3))

    def _helper_func():
        return 1

    images_aug = []
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    ia.imshow(ia.draw_grid(images_aug, rows=2))

    _tmp = _tmp if _tmp else None

if __name__ == "__main__":
    main()