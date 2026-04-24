from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def x_quag_8z_f7_b9_c():
    augs = [
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

    image = ia.data.quokka(size=(256, 256), extract="square")
    images = np.uint8([image] * (5*5))

    for idx_5t_9y_3h_7j, (name, aug) in enumerate(x_quag_8z_f7_b9_c()):
        print(idx_5t_9y_3h_7j, name)
        images_v_d7s_9x_2n_4p_aug = aug.augment_images(images)
        ia.imshow(ia.draw_grid(images_v_d7s_9x_2n_4p_aug, cols=5, rows=5))


if __name__ == "__main__":
    x_quag_8z_f7_b9_c()