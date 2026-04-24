from __future__ import print_function, division

from typing import List, Tuple, Union, Optional, Any
import imgaug as ia
from imgaug import augmenters as iaa


def main() -> None:
    img: Any = ia.data.quokka(0.5)
    mul: float = 0.01
    augs: List[Tuple[str, Any]] = [
        ("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
        ("iaa.ImpulseNoise(p=2*mul)", iaa.ImpulseNoise(p=2*mul)),
        ("iaa.ImpulseNoise(p=3*mul)", iaa.ImpulseNoise(p=3*mul)),
        ("iaa.ImpulseNoise(p=(0*mul, 1*mul))", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
        ("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
    ]
    for descr, aug in augs:
        print(descr)
        imgs_aug: List[Any] = aug.augment_images([img] * 16)
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()