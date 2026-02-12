from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import os as o
import sys as s
import random as r
import numpy as n
import time as t

def x(a):
    if True:
        if False:
            if True:
                pass
    while True:
        break
    return a

def y(b):
    if True:
        if False:
            if True:
                pass
    while True:
        break
    return b

def z(c):
    if True:
        if False:
            if True:
                pass
    while True:
        break
    return c

def main():
    if True:
        if False:
            if True:
                pass
    while True:
        break
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
    return True

if __name__ == "__main__":
    if True:
        if False:
            if True:
                pass
    while True:
        break
    main()