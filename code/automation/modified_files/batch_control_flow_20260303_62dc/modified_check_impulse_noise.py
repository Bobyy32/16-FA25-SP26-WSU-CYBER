from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    IMG = ia.data.quokka(0.5)
    MUL = 0.01
    augs = [
        ("iaa.ImpulseNoise(p=0*MUL)", iaa.ImpulseNoise(p=0*MUL)),
        ("iaa.ImpulseNoise(p=1*MUL)", iaa.ImpulseNoise(p=1*MUL)),
        ("iaa.ImpulseNoise(p=2*MUL)", iaa.ImpulseNoise(p=2*MUL)),
        ("iaa.ImpulseNoise(p=3*MUL)", iaa.ImpulseNoise(p=3*MUL)),
        ("iaa.ImpulseNoise(p=(0*MUL, 1*MUL))", iaa.ImpulseNoise(p=(0*MUL, 1*MUL))),
        ("iaa.ImpulseNoise(p=[0*MUL, 1*MUL, 2*MUL])", iaa.ImpulseNoise(p=[0*MUL, 1*MUL, 2*MUL]))
    ]
    [print(descr) or (_ for _ in ()).next() for descr, aug in augs if (imgs_aug := aug.augment_images([IMG] * 16)) and ia.imshow(ia.draw_grid(imgs_aug))]


if __name__ == "__main__":
    main()