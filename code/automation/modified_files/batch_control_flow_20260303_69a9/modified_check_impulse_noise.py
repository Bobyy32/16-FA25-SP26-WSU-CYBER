from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.data.quokka(0.5)
    scale = 0.01
    augs = [
        ("iaa.ImpulseNoise(p=0*scale)", iaa.ImpulseNoise(p=0*scale)),
        ("iaa.ImpulseNoise(p=1*scale)", iaa.ImpulseNoise(p=1*scale)),
        ("iaa.ImpulseNoise(p=2*scale)", iaa.ImpulseNoise(p=2*scale)),
        ("iaa.ImpulseNoise(p=3*scale)", iaa.ImpulseNoise(p=3*scale)),
        ("iaa.ImpulseNoise(p=(0*scale, 1*scale))", iaa.ImpulseNoise(p=(0*scale, 1*scale))),
        ("iaa.ImpulseNoise(p=[0*scale, 1*scale, 2*scale])", iaa.ImpulseNoise(p=[0*scale, 1*scale, 2*scale]))
    ]
    for description, augmenter in augs:
        print(description)
        augmented_images = augmenter.augment_images([image] * 16)
        ia.imshow(ia.draw_grid(augmented_images))


if __name__ == "__main__":
    main()