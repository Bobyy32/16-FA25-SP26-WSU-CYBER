from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    quokka_img = ia.quokka_square((128, 128))
    augmented_images = []

    for hue_sat_factor in np.linspace(0.0, 2.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(hue_sat_factor)
        augmented_img = augmenter.augment_image(quokka_img)
        augmented_images.append(augmented_img)

    for hue_factor in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_hue=hue_factor)
        augmented_img = augmenter.augment_image(quokka_img)
        augmented_images.append(augmented_img)

    for saturation_factor in np.linspace(0.0, 5.0, 10):
        augmenter = iaa.MultiplyHueAndSaturation(mul_saturation=saturation_factor)
        augmented_img = augmenter.augment_image(quokka_img)
        augmented_images.append(augmented_img)

    ia.imshow(ia.draw_grid(augmented_images, rows=3))

    augmented_images = []
    augmented_images.extend(iaa.MultiplyHue().augment_images([quokka_img] * 10))
    augmented_images.extend(iaa.MultiplySaturation().augment_images([quokka_img] * 10))
    ia.imshow(ia.draw_grid(augmented_images, rows=2))


if __name__ == "__main__":
    main()