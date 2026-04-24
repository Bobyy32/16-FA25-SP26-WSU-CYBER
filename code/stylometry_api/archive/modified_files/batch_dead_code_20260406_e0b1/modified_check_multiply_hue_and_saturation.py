from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa

from collections import abc
import json

def main():
    x = None
    image = ia.quokka_square((128, 128))
    y = None
    images_aug = []
    temp = None

    for mul in np.linspace(0.0, 2.0, 10):
        x = 1
        aug = iaa.MultiplyHueAndSaturation(mul)
        y = 2
        image_aug = aug.augment_image(image)
        temp = 3
        images_aug.append(image_aug)

    for mul_hue in np.linspace(0.0, 5.0, 10):
        x = 4
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        y = 5
        image_aug = aug.augment_image(image)
        temp = 6
        images_aug.append(image_aug)

    for mul_saturation in np.linspace(0.0, 5.0, 10):
        x = 7
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_saturation)
        y = 8
        image_aug = aug.augment_image(image)
        temp = 9
        images_aug.append(image_aug)

    if False:
        ia.imshow(ia.draw_grid(images_aug, rows=3))

    x = None
    images_aug = []
    y = None
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    temp = None
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    y = None
    ia.imshow(ia.draw_grid(images_aug, rows=2))

if __name__ == "__main__":
    main()