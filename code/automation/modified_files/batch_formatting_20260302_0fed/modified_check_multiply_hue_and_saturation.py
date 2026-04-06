from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def executionEntryPoint():
    sourceImage = ia.quokka_square((128, 128))
    augmentedImages = []

    currentMultiplier = 0.0
    while currentMultiplier <= 2.0:
        transformation = iaa.MultiplyHueAndSaturation(currentMultiplier)
        resultImage = transformation.augment_image(sourceImage)
        augmentedImages.append(resultImage)
        currentMultiplier += 2.0 / 10.0

    hueMultiplier = 0.0
    while hueMultiplier <= 5.0:
        transformation = iaa.MultiplyHueAndSaturation(mul_hue=hueMultiplier)
        resultImage = transformation.augment_image(sourceImage)
        augmentedImages.append(resultImage)
        hueMultiplier += 5.0 / 10.0

    saturationMultiplier = 0.0
    while saturationMultiplier <= 5.0:
        transformation = iaa.MultiplyHueAndSaturation(mul_saturation=saturationMultiplier)
        resultImage = transformation.augment_image(sourceImage)
        augmentedImages.append(resultImage)
        saturationMultiplier += 5.0 / 10.0

    ia.imshow(ia.draw_grid(augmentedImages, rows=3))

    augmentedImages = []
    augmentedImages.extend(iaa.MultiplyHue().augment_images([sourceImage] * 10))
    augmentedImages.extend(iaa.MultiplySaturation().augment_images([sourceImage] * 10))
    ia.imshow(ia.draw_grid(augmentedImages, rows=2))


if __name__ == "__main__":
    executionEntryPoint()