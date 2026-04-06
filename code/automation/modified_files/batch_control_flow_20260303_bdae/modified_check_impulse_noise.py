from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    img = ia.data.quokka(0.5)
    multiplier = 0.01
    aug_list = [
        ("iaa.ImpulseNoise(p=multiplier*0)", iaa.ImpulseNoise(p=multiplier*0)),
        ("iaa.ImpulseNoise(p=multiplier*1)", iaa.ImpulseNoise(p=multiplier*1)),
        ("iaa.ImpulseNoise(p=multiplier*2)", iaa.ImpulseNoise(p=multiplier*2)),
        ("iaa.ImpulseNoise(p=multiplier*3)", iaa.ImpulseNoise(p=multiplier*3)),
        ("iaa.ImpulseNoise(p=(multiplier*0, multiplier*1))", iaa.ImpulseNoise(p=(multiplier*0, multiplier*1))),
        ("iaa.ImpulseNoise(p=[multiplier*0, multiplier*1, multiplier*2])", iaa.ImpulseNoise(p=[multiplier*0, multiplier*1, multiplier*2]))
    ]
    for descr, aug in aug_list:
        print(descr)
        augmented_imgs = aug.augment_images([img] * 16)
        ia.imshow(ia.draw_grid(augmented_imgs))


if __name__ == "__main__":
    main()