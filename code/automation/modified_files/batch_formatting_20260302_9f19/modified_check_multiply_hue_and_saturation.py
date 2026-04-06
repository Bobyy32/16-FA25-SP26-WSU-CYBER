from __future__ import print_function, division

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    input_image = ia.quokka_square((128, 128))
    augmented_images = []

    for multiplier in np.linspace(0.0, 2.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(multiplier)
        augmented_image = augmenter.augment_image(input_image)
        augmented_images.append(augmented_image)

    for hue_multiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_hue=hue_multiplier)
        augmented_image = augmenter.augment_image(input_image)
        augmented_images.append(augmented_image)

    for saturation_multiplier in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_saturation=saturation_multiplier)
        augmented_image = augmenter.augment_image(input_image)
        augmented_images.append(augmented_image)

    ia.imshow(ia.draw_grid(augmented_images, rows=3))

    augmented_images = []
    augmented_images.extend(iaa.MultiplyHue().augment_images([input_image] * 10))
    augmented_images.extend(iaa.MultiplySaturation().augment_images([input_image] * 10))
    ia.imshow(ia.draw_grid(augmented_images, rows=2))


if __name__ == "__main__":
    main()