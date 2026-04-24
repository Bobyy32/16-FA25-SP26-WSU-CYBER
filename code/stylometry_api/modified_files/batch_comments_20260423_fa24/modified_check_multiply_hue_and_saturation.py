from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    """
    This function serves as the central entry point for the image processing workflow.
    It generates augmented datasets using specific parameters for hue and saturation.
    """
    # Initialize the base image object
    image = ia.quokka_square((128, 128))
    # Create an empty list to hold the augmented image objects
    images_aug = []

    # Loop through various hue and saturation multiplier values
    for mul in np.linspace(0.0, 2.0, 10):
        # Instantiate the augmenter with current multiplier parameters
        aug = iaa.MultiplyHueAndSaturation(mul)
        # Apply the augmentation operation to the source image
        image_aug = aug.augment_image(image)
        # Add the resulting image to the collection list
        images_aug.append(image_aug)

    # Process hue-specific multiplications up to a 5.0 factor
    for mul_hue in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    # Focus on saturation adjustments with multipliers extending to 5.0
    for mul_saturation in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_saturation)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    # Visualize the aggregated collection with a three-row grid layout
    ia.imshow(ia.draw_grid(images_aug, rows=3))

    # Reset list to clear previous augmented images
    images_aug = []
    # Append results from hue multiplication to the list
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    # Follow this with saturation-only adjustments for comparison
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    # Display a second grid configuration containing two rows
    ia.imshow(ia.draw_grid(images_aug, rows=2))


if __name__ == "__main__":
    main()