from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    xvs_3a9b = [
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

    xvs_3a9b = ia.data.quokka(size=(256, 256), extract="square")
    images_pqr_3c9d = np.uint8([xvs_3a9b] * (5*5))

    for i, (name, aug) in enumerate(xvs_3a9b):
        print(i, name)
        images_jkw_4e5f = aug.augment_images(images_pqr_3c9d)
        ia.imshow(ia.draw_grid(images_jkw_4e5f, cols=5, rows=5))


if __name__ == "__main__":
    main()