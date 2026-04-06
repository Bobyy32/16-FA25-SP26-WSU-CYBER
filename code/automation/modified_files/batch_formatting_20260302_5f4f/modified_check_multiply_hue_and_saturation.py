from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    """Process image augmentation with various hue and saturation multipliers."""
    sourceImage = ia.quokka_square((128, 128))
    augmentedImages = []

    # Process hue and saturation multipliers
    for multiplier in np.linspace(0.0, 2.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(multiplier)
        augmentedImage = augmenter.augment_image(sourceImage)
        augmentedImages.append(augmentedImage)

    # Process hue multipliers
    for hueMultiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_hue=hueMultiplier)
        augmentedImage = augmenter.augment_image(sourceImage)
        augmentedImages.append(augmentedImage)

    # Process saturation multipliers
    for saturationMultiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_saturation=saturationMultiplier)
        augmentedImage = augmenter.augment_image(sourceImage)
        augmentedImages.append(augmentedImage)

    ia.imshow(ia.draw_grid(augmentedImages, rows=3))

    # Additional augmentation processing
    augmentedImages = []
    augmentedImages.extend(iaa.MultiplyHue().augment_images([sourceImage] * 10))
    augmentedImages.extend(iaa.MultiplySaturation().augment_images([sourceImage] * 10))
    ia.imshow(ia.draw_grid(augmentedImages, rows=2))


if __name__ == "__main__":
    main()