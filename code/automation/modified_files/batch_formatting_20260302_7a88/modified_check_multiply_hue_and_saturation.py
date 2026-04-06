from __future__ import print_function, division

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def execute_program():
    """Process image augmentation with various hue and saturation multipliers."""
    source_picture = ia.quokka_square((128, 128))
    augmented_pictures = []

    # Process hue and saturation multipliers
    for multiplier in np.linspace(0.0, 2.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(multiplier)
        augmented_picture = augmenter.augment_image(source_picture)
        augmented_pictures.append(augmented_picture)

    # Process hue multipliers
    for hue_multiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_hue=hue_multiplier)
        augmented_picture = augmenter.augment_image(source_picture)
        augmented_pictures.append(augmented_picture)

    # Process saturation multipliers
    for saturation_multiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_saturation=saturation_multiplier)
        augmented_picture = augmenter.augment_image(source_picture)
        augmented_pictures.append(augmented_picture)

    ia.imshow(ia.draw_grid(augmented_pictures, rows=3))

    # Additional augmentation operations
    augmented_pictures = []
    augmented_pictures.extend(iaa.MultiplyHue().augment_images([source_picture] * 10))
    augmented_pictures.extend(iaa.MultiplySaturation().augment_images([source_picture] * 10))
    ia.imshow(ia.draw_grid(augmented_pictures, rows=2))


if __name__ == "__main__":
    execute_program()