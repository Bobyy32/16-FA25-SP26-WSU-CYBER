from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa
from typing import Union, Optional


def main() -> None:
    img: ia.data.Image = ia.data.quokka(0.5)
    mul: float = 0.01
    augs: list[tuple[str, Union[ia.augmenters.BaseAugmenter, ...]]] = [
        ("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
        ("iaa.ImpulseNoise(p=2*mul)", iaa.ImpulseNoise(p=2*mul)),
        ("iaa.ImpulseNoise(p=3*mul)", iaa.ImpulseNoise(p=3*mul)),
        ("iaa.ImpulseNoise(p=(0*mul, 1*mul))", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
        ("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
    ]
    for descr: str, aug: Union[ia.augmenters.BaseAugmenter, ...] in augs:
        print(descr)
        imgs_aug: list[ia.data.Image] = aug.augment_images([img] * 16)
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()