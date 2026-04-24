from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    # Create a square test image using quokka generator
    image = ia.quokka_square((128, 128))
    
    # Container to store augmented images for visualization
    images_aug = []

    # Apply MultiplyHueAndSaturation across hue values from 0 to 2
    for multiplicative_factor in np.linspace(0.0, 2.0, 10):
        hue_sat_aug = iaa.MultiplyHueAndSaturation(multiplicative_factor)
        augmented_image = hue_sat_aug.augment_image(image)
        images_aug.append(augmented_image)

    # Apply MultiplyHueAndSaturation across hue values from 0 to 5
    for hue_multiplier in np.linspace(0.0, 5.0, 10):
        hue_sat_aug = iaa.MultiplyHueAndSaturation(hue_multiplier=hue_multiplier)
        augmented_image = hue_sat_aug.augment_image(image)
        images_aug.append(augmented_image)

    # Apply MultiplyHueAndSaturation across saturation values from 0 to 5
    for saturation_multiplier in np.linspace(0.0, 5.0, 10):
        hue_sat_aug = iaa.MultiplyHueAndSaturation(saturation_multiplier=saturation_multiplier)
        augmented_image = hue_sat_aug.augment_image(image)
        images_aug.append(augmented_image)

    # Display augmented images in a 3-row grid with draw_grid visualization
    ia.imshow(ia.draw_grid(images_aug, rows=3))

    # Prepare new augmented image set for second visualization display
    images_aug = []
    
    # Extend with hue-only multiplicative variations on a sample batch
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    
    # Extend with saturation-only multiplicative variations on the same batch
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    
    # Present the final collection in a 2-row grid
    ia.imshow(ia.draw_grid(images_aug, rows=2))


if __name__ == "__main__":
    main()