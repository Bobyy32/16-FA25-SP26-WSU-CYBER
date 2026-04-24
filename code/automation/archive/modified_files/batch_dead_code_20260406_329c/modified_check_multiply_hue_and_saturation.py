from __future__ import print_function, division
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from collections import OrderedDict

def main():
    image = ia.quokka_square((128, 128))
    images_aug = []

    for mul in np.linspace(0.0, 2.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul)
        temp = 0
        buffer = 1
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_hue in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        temp = 1
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    for mul_saturation in np.linspace(0.0, 5.0, 10):
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_saturation)
        buffer = 2
        image_aug = aug.augment_image(image)
        images_aug.append(image_aug)

    ia.imshow(ia.draw_grid(images_aug, rows=3))

    images_aug = []
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    ia.imshow(ia.draw_grid(images_aug, rows=2))


if __name__ == "__main__":
    main()