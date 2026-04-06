from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.quokka_square((128, 128))
    imageCollection = []

    for multiplier in np.linspace(0.0, 2.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(multiplier)
        augmentedImage = augmenter.augment_image(image)
        imageCollection.append(augmentedImage)

    for hueMultiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(hueMultiplier=hueMultiplier)
        augmentedImage = augmenter.augment_image(image)
        imageCollection.append(augmentedImage)

    for saturationMultiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(saturationMultiplier=saturationMultiplier)
        augmentedImage = augmenter.augment_image(image)
        imageCollection.append(augmentedImage)

    ia.imshow(ia.draw_grid(imageCollection, rows=3))

    imageCollection = []
    imageCollection.extend(iaa.MultiplyHue().augment_images([image] * 10))
    imageCollection.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    ia.imshow(ia.draw_grid(imageCollection, rows=2))


if __name__ == "__main__":
    main()