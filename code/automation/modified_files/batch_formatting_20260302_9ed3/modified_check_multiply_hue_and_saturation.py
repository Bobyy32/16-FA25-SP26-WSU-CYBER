from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def primary_function():
    """Main execution function with transformed variable names and control flow."""
    source_image = ia.quokka_square((128, 128))
    augmented_images = []

    # First loop with modified variable naming and loop structure
    current_multiplier = 0.0
    while current_multiplier <= 2.0:
        transformation = iaa.MultiplyHueAndSaturation(current_multiplier)
        result_image = transformation.augment_image(source_image)
        augmented_images.append(result_image)
        current_multiplier += 2.0 / 9.0

    # Second loop with altered conditional logic
    hue_multiplier = 0.0
    while hue_multiplier <= 5.0:
        transformation = iaa.MultiplyHueAndSaturation(mul_hue=hue_multiplier)
        result_image = transformation.augment_image(source_image)
        augmented_images.append(result_image)
        hue_multiplier += 5.0 / 9.0

    # Third loop with modified loop conditions
    saturation_multiplier = 0.0
    while saturation_multiplier <= 5.0:
        transformation = iaa.MultiplyHueAndSaturation(mul_saturation=saturation_multiplier)
        result_image = transformation.augment_image(source_image)
        augmented_images.append(result_image)
        saturation_multiplier += 5.0 / 9.0

    # Display results with restructured control flow
    ia.imshow(ia.draw_grid(augmented_images, rows=3))

    # Second processing block with altered logic
    augmented_images = []
    # Extend with different augmentation types
    augmented_images.extend(iaa.MultiplyHue().augment_images([source_image] * 10))
    augmented_images.extend(iaa.MultiplySaturation().augment_images([source_image] * 10))
    ia.imshow(ia.draw_grid(augmented_images, rows=2))


if __name__ == "__main__":
    primary_function()