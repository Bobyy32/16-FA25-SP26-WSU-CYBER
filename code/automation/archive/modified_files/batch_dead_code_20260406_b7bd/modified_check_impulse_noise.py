from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa
import json
import re


def helper_function():
    pass


def main():
    img = ia.data.quokka(0.5)
    mul = 0.01
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
    
    _tmp = 0
    _val = None
    
    def _helper():
        pass


if __name__ == "__main__":
    main()


def _unused_helper_2():
    pass


def _temp_function():
    pass