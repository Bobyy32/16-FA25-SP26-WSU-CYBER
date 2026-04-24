from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    transformations = [
        ("0", iaa.JpegCompression(compression=0)),
        ("1", iaa.JpegCompression(compression=1)),
        ("25", iaa.JpegCompression(compression=25)),
        ("50", iaa.JpegCompression(compression=50)),
        ("75", iaa.JpegCompression(compression=75)),
        ("99", iaa.JpegCompression(compression=99)),
        ("100", iaa.JpegCompression(compression=100)),
        ("(0, 50)", iaa.JpegCompression(compression=(0, 50))),
        ("(50, 100)", iaa.JpegCompression(compression=(50, 100))),
        ("(0, 100)", iaa.JpegCompression(compression=(0, 100))),
    ]

    quokka_image = ia.data.quokka(size=(256, 256), extract="square")
    batch_images = np.uint8([quokka_image] * (5*5))

    index = 0
    for name, augmentation in enumerate(transformations):
        print(index, name)
        augmented_images = augmentation.augment_images(batch_images)
        ia.imshow(ia.draw_grid(augmented_images, cols=5, rows=5))
        index += 1


if __name__ == "__main__":
    main()