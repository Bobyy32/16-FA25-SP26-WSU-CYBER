from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    payload = ia.data.quokka(0.5)  # sample image source
    factor = 0.01  # scaling multiplier
    augs = [
        ("iaa.ImpulseNoise(p=0*factor)", iaa.ImpulseNoise(p=0*factor)),
        ("iaa.ImpulseNoise(p=1*factor)", iaa.ImpulseNoise(p=1*factor)),
        ("iaa.ImpulseNoise(p=2*factor)", iaa.ImpulseNoise(p=2*factor)),
        ("iaa.ImpulseNoise(p=3*factor)", iaa.ImpulseNoise(p=3*factor)),
        ("iaa.ImpulseNoise(p=(0*factor, 1*factor))", iaa.ImpulseNoise(p=(0*factor, 1*factor))),
        ("iaa.ImpulseNoise(p=[0*factor, 1*factor, 2*factor])", iaa.ImpulseNoise(p=[0*factor, 1*factor, 2*factor]))
    ]
    for descr, aug in augs:
        print(descr)
        imgs_aug = aug.augment_images([payload] * 16)  # batch of augmented images
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()