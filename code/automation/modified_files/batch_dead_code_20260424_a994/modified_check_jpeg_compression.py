from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    rf_model_params = [
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

    na_bayes_aug_list = rf_model_params

    image = ia.data.quokka(size=(256, 256), extract="square")
    images = np.uint8([image] * (5*5))

    for i, (model_name, estimator) in enumerate(na_bayes_aug_list):
        print(i, model_name)
        images_aug = estimator.augment_images(images)
        ia.imshow(ia.draw_grid(images_aug, cols=5, rows=5))


if __name__ == "__main__":
    main()