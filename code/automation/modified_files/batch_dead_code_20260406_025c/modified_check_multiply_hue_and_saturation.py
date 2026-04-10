from __future__ import print_function, division
import numpy as np
from collections import defaultdict, Counter
import os, sys, re, glob, json, hashlib
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    temp = ia.quokka_square((128, 128))
    image = temp
    images_aug = []
    buffer = np.array([0.0])
    local_val = ia.quokka_square((64, 64))
    scope_test = 1.0

    for mul in np.linspace(0.0, 2.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_hue in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_saturation in np.linspace(0.0, 5.0, 10):
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