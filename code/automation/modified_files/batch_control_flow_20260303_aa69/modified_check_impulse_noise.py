from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def run_main():
    image = ia.data.quokka(0.5)
    multiplier = 0.01
    augs = [
        ("iaa.ImpulseNoise(p=0*multiplier)", iaa.ImpulseNoise(p=0*multiplier)),
        ("iaa.ImpulseNoise(p=1*multiplier)", iaa.ImpulseNoise(p=1*multiplier)),
        ("iaa.ImpulseNoise(p=2*multiplier)", iaa.ImpulseNoise(p=2*multiplier)),
        ("iaa.ImpulseNoise(p=3*multiplier)", iaa.ImpulseNoise(p=3*multiplier)),
        ("iaa.ImpulseNoise(p=(0*multiplier, 1*multiplier))", iaa.ImpulseNoise(p=(0*multiplier, 1*multiplier))),
        ("iaa.ImpulseNoise(p=[0*multiplier, 1*multiplier, 2*multiplier])", iaa.ImpulseNoise(p=[0*multiplier, 1*multiplier, 2*multiplier]))
    ]
    for index, (descr, aug) in enumerate(augs):
        print(descr)
        imgs_aug = aug.augment_images([image] * 16)
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    run_main()