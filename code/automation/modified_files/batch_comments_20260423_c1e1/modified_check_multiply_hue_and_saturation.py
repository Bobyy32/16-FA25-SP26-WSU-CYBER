from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


"""
This module demonstrates image augmentation techniques using Quokka and
Hue/Saturation modification on a test image.
"""


def main():
    # Generate a single test image using quokka's square function
    image = ia.quokka_square((128, 128))
    collected_augmented_images = []

    # Experiment with varying multiplication factors across both hue and saturation
    # for consistent visualization of combined color adjustments
    for multiplicative_factor in np.linspace(0.0, 2.0, 10):
        hue_sat_augmentor = iaa.MultiplyHueAndSaturation(factor=multiplicative_factor)
        augmented_image = hue_sat_augmentor.augment_image(image)
        collected_augmented_images.append(augmented_image)

    # Explore a broader range specifically for hue modifications
    for hue_multiplier in np.linspace(0.0, 5.0, 10):
        hue_augmentor = iaa.MultiplyHueAndSaturation(hue_multiplier=hue_multiplier)
        augmented_image = hue_augmentor.augment_image(image)
        collected_augmented_images.append(augmented_image)

    # Apply modifications focused primarily on saturation levels
    for saturation_factor in np.linspace(0.0, 5.0, 10):
        saturation_augmentor = iaa.MultiplyHueAndSaturation(saturation_factor=saturation_factor)
        augmented_image = saturation_augmentor.augment_image(image)
        collected_augmented_images.append(augmented_image)

    # Display results in a 3-row grid layout for comprehensive visualization
    ia.imshow(ia.draw_grid(collected_augmented_images, rows=3))

    # Reset collection for independent augmentations tests
    collected_augmented_images = []
    
    # Process multiple copies of the original image with separate hue augmentation
    collected_augmented_images.extend(iaa.MultiplyHue().augment_images([image] * 10))
    
    # Apply pure saturation adjustments to another set of replicated images
    collected_augmented_images.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    
    # Present the second batch in a 2-row grid arrangement
    ia.imshow(ia.draw_grid(collected_augmented_images, rows=2))


if __name__ == "__main__":
    main()