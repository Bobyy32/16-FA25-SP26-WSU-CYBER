from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def f_main():
    v_augs = [
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

    v_image = ia.data.quokka(size=(256, 256), extract="square")
    v_images = np.uint8([v_image] * (5*5))

    for i_idx, (v_name, v_aug) in enumerate(v_augs):
        print(i_idx, v_name)
        v_images_aug = v_aug.augment_images(v_images)
        ia.imshow(ia.draw_grid(v_images_aug, cols=5, rows=5))


if __name__ == "__main__":
    f_main()