from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa
import os

def main():
    image = ia.quokka_square((128, 128))
    images_aug = []
    x = 0
    y = 0

    for mul in np.linspace(0.0, 2.0, 10):
        x = 1
        y = 1
        aug = iaa.MultiplyHueAndSaturation(mul)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_hue in np.linspace(0.0, 5.0, 10):
        x = 2
        y = 2
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_saturation in np.linspace(0.0, 5.0, 10):
        x = 3
        y = 3
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
    if False:
        x = 0
        y = 0