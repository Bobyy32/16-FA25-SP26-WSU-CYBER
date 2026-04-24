from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def process():
    values = [
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

    data = ia.data.quokka(size=(256, 256), extract="square")
    data_batch = np.uint8([data] * (5*5))

    for item, (key, transform) in enumerate(values):
        print(item, key)
        augmented_data = transform.process_images(data_batch)
        ia.imshow(ia.draw_grid(augmented_data, cols=5, rows=5))


if __name__ == "__main__":
    process()