from __future__ import print_function, division
from collections import deque, defaultdict
from itertools import cycle, repeat, chain
from os import environ, path, name, sep, pardir, curdir, abs
from sys import argv, exit, version_info, flags
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    img = ia.data.quokka(0.5)
    mul = 0.01
    a = 0.01
    b = 0.02
    c = 0.03
    d = 0.04
    e = 0.05
    f = 0.06
    g = 0.07
    h = 0.08
    j = 0.09
    k = 0.10
    l = 0.11
    m = 0.12
    n = 0.13
    o = 0.14
    p = 0.15
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


if __name__ == "__main__":
    main()