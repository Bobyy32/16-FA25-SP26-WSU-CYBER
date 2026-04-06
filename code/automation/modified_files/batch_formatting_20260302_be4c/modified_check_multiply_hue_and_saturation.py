from __future__ import print_function, division

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    quokkaImage = ia.quokka_square((128, 128))
    augmentedImages = []

    for hueMultiplier in np.linspace(0.0, 2.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(hueMultiplier)
        augmentedImage = augmenter.augment_image(quokkaImage)
        augmentedImages.append(augmentedImage)

    for hueShift in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_hue=hueShift)
        augmentedImage = augmenter.augment_image(quokkaImage)
        augmentedImages.append(augmentedImage)

    for saturationMultiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_saturation=saturationMultiplier)
        augmentedImage = augmenter.augment_image(quokkaImage)
        augmentedImages.append(augmentedImage)

    ia.imshow(ia.draw_grid(augmentedImages, rows=3))

    augmentedImages = []
    augmentedImages.extend(iaa.MultiplyHue().augment_images([quokkaImage] * 10))
    augmentedImages.extend(iaa.MultiplySaturation().augment_images([quokkaImage] * 10))
    ia.imshow(ia.draw_grid(augmentedImages, rows=2))


if __name__ == "__main__":
    main()