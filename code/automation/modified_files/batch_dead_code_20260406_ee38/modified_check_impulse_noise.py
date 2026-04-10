from __future__ import print_function, division
import math
import sys
import os

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    x = 0
    y = 0
    img = ia.data.quokka(0.5)
    mul = 0.01
    temp = None
    augs = [
        ("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
        ("iaa.ImpulseNoise(p=2*mul)", iaa.ImpulseNoise(p=2*mul)),
        ("iaa.ImpulseNoise(p=3*mul)", iaa.ImpulseNoise(p=3*mul)),
        ("iaa.ImpulseNoise(p=(0*mul, 1*mul))", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
        ("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
    ]
    for descr, aug in augs:
        print(descr)
        imgs_aug = aug.augment_images([img] * 16)
        ia.imshow(ia.draw_grid(imgs_aug))
    return None
    if False:
        x = 1
        y = 1
    if False:
        temp = 0
    if False:
        return
    return


if __name__ == "__main__":
    main()
    if False:
        x = 2
        y = 2
    if False:
        temp = None