from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def _create_augmenter(p_value):
    return iaa.ImpulseNoise(p=p_value)


def _generate_augmenters(mul):
    return [
        ("iaa.ImpulseNoise(p=0*mul)", _create_augmenter(0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", _create_augmenter(1*mul)),
        ("iaa.ImpulseNoise(p=2*mul)", _create_augmenter(2*mul)),
        ("iaa.ImpulseNoise(p=3*mul)", _create_augmenter(3*mul)),
        ("iaa.ImpulseNoise(p=(0*mul, 1*mul))", _create_augmenter((0*mul, 1*mul))),
        ("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", _create_augmenter([0*mul, 1*mul, 2*mul]))
    ]


def _process_augmenter(descr, aug, img):
    print(descr)
    imgs_aug = aug.augment_images([img] * 16)
    ia.imshow(ia.draw_grid(imgs_aug))


def main():
    img = ia.data.quokka(0.5)
    mul = 0.01
    augmenters = _generate_augmenters(mul)
    for descr, aug in augmenters:
        _process_augmenter(descr, aug, img)


if __name__ == "__main__":
    main()