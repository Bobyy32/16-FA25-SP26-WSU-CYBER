from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    augmenters_config = [
        ("zero", iaa.JpegCompression(compression=0)),
        ("one", iaa.JpegCompression(compression=1)),
        ("twenty-five", iaa.JpegCompression(compression=25)),
        ("fifty", iaa.JpegCompression(compression=50)),
        ("seventy-five", iaa.JpegCompression(compression=75)),
        ("ninety-nine", iaa.JpegCompression(compression=99)),
        ("one-hundred", iaa.JpegCompression(compression=100)),
        ("zero-to-fifty", iaa.JpegCompression(compression=(0, 50))),
        ("fifty-to-one-hundred", iaa.JpegCompression(compression=(50, 100))),
        ("zero-to-one-hundred", iaa.JpegCompression(compression=(0, 100))),
    ]

    image = ia.data.quokka(size=(256, 256), extract="square")
    images_array = np.uint8([image] * (5*5))

    for i, (augmentation_name, augmentation) in enumerate(augmenters_config):
        print(i, augmentation_name)
        augmented_images = augmentation.augment_images(images_array)
        ia.imshow(ia.draw_grid(augmented_images, cols=5, rows=5))


if __name__ == "__main__":
    main()