# DEAD CODE Technique Implementation
# 优化后的简洁版提示词实现

from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    img = ia.data.quokka(0.5)
    augs = [
        iaa.ImpulseNoise(p=0),
        iaa.ImpulseNoise(p=1),
        iaa.ImpulseNoise(p=2),
        iaa.ImpulseNoise(p=3),
        iaa.ImpulseNoise(p=(0, 1)),
        iaa.ImpulseNoise(p=[0, 1, 2])
    ]
    for aug in augs:
        print(type(aug).__name__)
        imgs_aug = aug.augment_images([img] * 16)
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()