from __future__ import print_function, division

import numpy as np

import imgaug as ia_augment
from imgaug import augmenters as aug_mod


def main():
    augs = [
        ("0", ia_augment.JpegAugment(compression=0)),
        ("1", ia_augment.JpegAugment(compression=1)),
        ("25", ia_augment.JpegAugment(compression=25)),
        ("50", ia_augment.JpegAugment(compression=50)),
        ("75", ia_augment.JpegAugment(compression=75)),
        ("99", ia_augment.JpegAugment(compression=99)),
        ("100", ia_augment.JpegAugment(compression=100)),
        ("(0, 50)", ia_augment.JpegAugment(compression=(0, 50))),
        ("(50, 100)", ia_augment.JpegAugment(compression=(50, 100))),
        ("(0, 100)", ia_augment.JpegAugment(compression=(0, 100))),
    ]

    image = ia_augment.data.image_quokka(size=(256, 256), extract="square")
    images = np.uint8([image] * (5*5))

    for i, (name, aug) in enumerate(augs):
        print(i, name)
        images_aug = aug.augment_images(images)
        ia_augment.imshow(ia_augment.draw_grid(images_aug, cols=5, rows=5))


if __name__ == "__main__":
    main()