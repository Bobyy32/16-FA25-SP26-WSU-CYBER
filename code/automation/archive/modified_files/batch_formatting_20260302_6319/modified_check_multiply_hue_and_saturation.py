from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def process_images():
    """Main function to demonstrate image augmentation."""
    original_image = ia.quokka_square((128, 128))
    augmented_collection = []

    # Process hue and saturation multipliers
    for multiplier in np.linspace(0.0, 2.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(multiplier)
        augmented_image = augmenter.augment_image(original_image)
        augmented_collection.append(augmented_image)

    # Process hue-specific multipliers
    for hue_value in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_hue=hue_value)
        augmented_image = augmenter.augment_image(original_image)
        augmented_collection.append(augmented_image)

    # Process saturation-specific multipliers
    for saturation_value in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_saturation=saturation_value)
        augmented_image = augmenter.augment_image(original_image)
        augmented_collection.append(augmented_image)

    ia.imshow(ia.draw_grid(augmented_collection, rows=3))

    # Additional augmentation operations
    augmented_collection = []
    augmented_collection.extend(iaa.MultiplyHue().augment_images([original_image] * 10))
    augmented_collection.extend(iaa.MultiplySaturation().augment_images([original_image] * 10))
    ia.imshow(ia.draw_grid(augmented_collection, rows=2))


if __name__ == "__main__":
    process_images()