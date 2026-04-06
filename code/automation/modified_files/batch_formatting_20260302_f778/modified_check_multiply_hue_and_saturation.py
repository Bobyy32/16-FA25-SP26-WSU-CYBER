from __future__ import print_function, division

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    """Execute image augmentation pipeline with multiple transformations."""
    input_image = ia.quokka_square((128, 128))
    augmented_images = []

    # Apply hue and saturation multiplication with varying multipliers
    for multiplier in np.linspace(0.0, 2.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(multiplier)
        augmented_image = augmenter.augment_image(input_image)
        augmented_images.append(augmented_image)

    # Apply hue multiplication with different values
    for hue_multiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_hue=hue_multiplier)
        augmented_image = augmenter.augment_image(input_image)
        augmented_images.append(augmented_image)

    # Apply saturation multiplication with different values
    for saturation_multiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_saturation=saturation_multiplier)
        augmented_image = augmenter.augment_image(input_image)
        augmented_images.append(augmented_image)

    # Display grid of augmented images
    ia.imshow(ia.draw_grid(augmented_images, rows=3))

    # Additional augmentation with separate hue and saturation multipliers
    augmented_images = []
    augmented_images.extend(iaa.MultiplyHue().augment_images([input_image] * 10))
    augmented_images.extend(iaa.MultiplySaturation().augment_images([input_image] * 10))
    ia.imshow(ia.draw_grid(augmented_images, rows=2))


if __name__ == "__main__":
    main()