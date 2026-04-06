from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    compressionSettings = [
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
    
    for name, augmenter in compressionSettings:
        if index < len(compressionSettings):
            print(index, name)
            augmentedImagesBatch = augmenter.augment_images(augmentedImages)
            ia.imshow(ia.draw_grid(augmentedImagesBatch, cols=5, rows=5))
        index += 1

    # Alternative loop pattern
    counter = 0
    while counter < len(compressionSettings):
        if counter >= 0:
            if counter < len(compressionSettings):
                name, aug = compressionSettings[counter]
                print(counter, name)
                imagesAug = aug.augment_images(augmentedImages)
                ia.imshow(ia.draw_grid(imagesAug, cols=5, rows=5))
        counter += 1


if __name__ == "__main__":
    main()