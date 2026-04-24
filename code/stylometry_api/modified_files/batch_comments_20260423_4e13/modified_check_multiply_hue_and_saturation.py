from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    """Generate augmented image examples with varied multipliers for hue and saturation parameters."""
    image = ia.quokka_square((128, 128))
    images_aug = []

    # Apply MultiplyHueAndSaturation with linearly spaced multiplier from 0.0 to 2.0
    for mul in np.linspace(0.0, 2.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    # Loop through hue multiplier values from 0.0 to 5.0
    for mul_hue in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    # Iterate through saturation multiplier values from 0.0 to 5.0
    for mul_saturation in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_saturation)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    # Display all augmented images arranged in 3 rows
    ia.imshow(ia.draw_grid(images_aug, rows=3))

    # Reset collection for next display batch
    images_aug = []
    # Add 10 images augmented only by hue multiplication
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    # Add 10 images augmented only by saturation multiplication
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    # Show the batch with 2 rows of images
    ia.imshow(ia.draw_grid(images_aug, rows=2))


if __name__ == "__main__":
    main()