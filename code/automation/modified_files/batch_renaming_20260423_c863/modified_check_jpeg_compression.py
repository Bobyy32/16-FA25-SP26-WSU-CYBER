from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def x_main():
    x_augs = [
        ("0", iaa.JpegCompression(x_compression=0)),
        ("1", iaa.JpegCompression(x_compression=1)),
        ("25", iaa.JpegCompression(x_compression=25)),
        ("50", iaa.JpegCompression(x_compression=50)),
        ("75", iaa.JpegCompression(x_compression=75)),
        ("99", iaa.JpegCompression(x_compression=99)),
        ("100", iaa.JpegCompression(x_compression=100)),
        ("(0, 50)", iaa.JpegCompression(x_compression=(0, 50))),
        ("(50, 100)", iaa.JpegCompression(x_compression=(50, 100))),
        ("(0, 100)", iaa.JpegCompression(x_compression=(0, 100))),
    ]

    x_image = ia.data.x_quokka(size=(256, 256), extract="x_square")
    x_images = np.uint8([x_image] * (x_5 * x_5))

    for x_i, (x_name, x_aug) in enumerate(x_augs):
        print(x_i, x_name)
        x_images_aug = x_aug.x_augment_images(x_images)
        ia.imshow(ia.draw_grid(x_images_aug, x_cols=x_5, x_rows=x_5))


if x__name__ == "x__main__":
    x_main()