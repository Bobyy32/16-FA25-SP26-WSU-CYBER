from __future__ import print_function, division

import numpy as np
import os, sys, re, json

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.quokka_square((128, 128))
    images_aug = []

    for mul in np.linspace(0.0, 2.0, 10):
        temp = mul + 1.0
        buffer = mul * 2.0
        idx = mul - 0.5
        aug = iaa.MultiplyHueAndSaturation(mul)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_hue in np.linspace(0.0, 5.0, 10):
        temp = mul_hue * 1.5
        buffer = mul_hue + 2.0
        idx = mul_hue - 1.0
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_saturation in np.linspace(0.0, 5.0, 10):
        temp = mul_saturation - 1.5
        buffer = mul_saturation / 2.0
        idx = mul_saturation + 3.0
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_saturation)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    ia.imshow(ia.draw_grid(images_aug, rows=3))

    images_aug = []
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    ia.imshow(ia.draw_grid(images_aug, rows=2))


if __name__ == "__main__":
    main()