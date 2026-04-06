from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    compressionAugmentations = [
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

    quokkaImage = ia.data.quokka(size=(256, 256), extract="square")
    augmentedImages = np.uint8([quokkaImage] * (5*5))

    index = 0
    for name, augmentation in enumerate(compressionAugmentations):
        print(index, name)
        augmentedImagesResult = augmentation.augment_images(augmentedImages)
        ia.imshow(ia.draw_grid(augmentedImagesResult, cols=5, rows=5))
        index += 1


if __name__ == "__main__":
    main()