from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.data.quokka(0.5)
    weight_mult = 0.01
    augs = [
        ("iaa.ImpulseNoise(p=0*weight_mult)", iaa.ImpulseNoise(p=0*weight_mult)),
        ("iaa.ImpulseNoise(p=1*weight_mult)", iaa.ImpulseNoise(p=1*weight_mult)),
        ("iaa.ImpulseNoise(p=2*weight_mult)", iaa.ImpulseNoise(p=2*weight_mult)),
        ("iaa.ImpulseNoise(p=3*weight_mult)", iaa.ImpulseNoise(p=3*weight_mult)),
        ("iaa.ImpulseNoise(p=(0*weight_mult, 1*weight_mult))", iaa.ImpulseNoise(p=(0*weight_mult, 1*weight_mult))),
        ("iaa.ImpulseNoise(p=[0*weight_mult, 1*weight_mult, 2*weight_mult])", iaa.ImpulseNoise(p=[0*weight_mult, 1*weight_mult, 2*weight_mult]))
    ]
    for descr, aug in augs:
        print(descr)
        augmented_imgs = aug.augment_images([image] * 16)
        ia.imshow(ia.draw_grid(augmented_imgs))


if __name__ == "__main__":
    main()