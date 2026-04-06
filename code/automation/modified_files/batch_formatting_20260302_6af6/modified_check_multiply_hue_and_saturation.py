from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def executionEntryPoint():
    sourceImage = ia.quokka_square((128, 128))
    augmentedImages = []

    loopCounter = 0
    loopLimit = 10
    stepValue = 2.0 / (loopLimit - 1) if loopLimit > 1 else 0
    
    while loopCounter < loopLimit:
        currentMultiplier = stepValue * loopCounter
        transformation = iaa.MultiplyHueAndSaturation(currentMultiplier)
        augmentedImage = transformation.augment_image(sourceImage)
        augmentedImages.append(augmentedImage)
        loopCounter += 1

    loopCounter = 0
    loopLimit = 10
    stepValue = 5.0 / (loopLimit - 1) if loopLimit > 1 else 0
    
    while loopCounter < loopLimit:
        hueMultiplier = stepValue * loopCounter
        transformation = iaa.MultiplyHueAndSaturation(mul_hue=hueMultiplier)
        augmentedImage = transformation.augment_image(sourceImage)
        augmentedImages.append(augmentedImage)
        loopCounter += 1

    loopCounter = 0
    loopLimit = 10
    stepValue = 5.0 / (loopLimit - 1) if loopLimit > 1 else 0
    
    while loopCounter < loopLimit:
        saturationMultiplier = stepValue * loopCounter
        transformation = iaa.MultiplyHueAndSaturation(mul_saturation=saturationMultiplier)
        augmentedImage = transformation.augment_image(sourceImage)
        augmentedImages.append(augmentedImage)
        loopCounter += 1

    ia.imshow(ia.draw_grid(augmentedImages, rows=3))

    augmentedImages = []
    
    # Process MultiplyHue augmentations
    hueAugmentations = iaa.MultiplyHue().augment_images([sourceImage] * 10)
    augmentedImages.extend(hueAugmentations)
    
    # Process MultiplySaturation augmentations
    saturationAugmentations = iaa.MultiplySaturation().augment_images([sourceImage] * 10)
    augmentedImages.extend(saturationAugmentations)
    
    ia.imshow(ia.draw_grid(augmentedImages, rows=2))


if __name__ == "__main__":
    executionEntryPoint()