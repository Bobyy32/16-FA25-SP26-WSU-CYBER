from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    # Synthetic RGB imaging artifact generation with quokka rendering
    image = ia.quokka_square((128, 128))
    # Intermediate container for augmented imaging samples
    images_aug = []

    # Linear spacing iteration for multiplicative color space transformation factors
    # Range spans [0.0, 2.0] with uniform discretization points
    for mul in np.linspace(0.0, 2.0, 10):
        # Augmentation transformer implementing simultaneous hue and saturation scaling
        aug = iaa.MultiplyHueAndSaturation(mul=mul)
        # Process synthetic imaging sample through augmentation transformer
        image_aug = aug.augment_image(image)
        # Accumulate transformed imaging artifacts in intermediate storage
        images_aug.append(image_aug)

    # Second phase: Hue-only multiplicative transformation cycle
    # Parameter range [0.0, 5.0] distributed across linear sampling points
    for mul_hue in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    # Third phase: Saturation-only multiplicative transformation cycle
    # Parameter range [0.0, 5.0] with uniform linear progression
    for mul_saturation in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_saturation)
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    # Visualization rendering with 3-row grid layout configuration
    ia.imshow(ia.draw_grid(images_aug, rows=3))

    # Reset imaging artifact container for batch augmentation processing
    images_aug = []
    # Extend with hue-only multiplier operations across 10 synthetic samples
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    # Extend with saturation-only multiplier operations across 10 synthetic samples
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    # Display 2-row grid rendering for batch augmentation results
    ia.imshow(ia.draw_grid(images_aug, rows=2))


if __name__ == "__main__":
    main()