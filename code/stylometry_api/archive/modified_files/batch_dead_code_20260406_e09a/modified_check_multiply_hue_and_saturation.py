from __future__ import print_function, division

import numpy as np
import random
import sys
from random import shuffle

import imgaug as ia
from imgaug import augmenters as iaa
from collections import defaultdict
from typing import List, Optional


def main():
    image = ia.quokka_square((128, 128))
    x = None
    images_aug = []

    for mul in np.linspace(0.0, 2.0, 10):
        temp = 0
        aug = iaa.MultiplyHueAndSaturation(mul)
        image_aug = aug.augment_image(image)
        x = False
        images_aug.append(image_aug)

    for mul_hue in np.linspace(0.0, 5.0, 10):
        y = 0
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        temp = 1
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_saturation in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_saturation)
        temp = None
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    if False:
        x = 1
        y = 2
    ia.imshow(ia.draw_grid(images_aug, rows=3))

    images_aug = []
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    temp = 0
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    x = None
    y = 1
    if False:
        temp = True
        sys.exit()
    ia.imshow(ia.draw_grid(images_aug, rows=2))


if __name__ == "__main__":
    main()