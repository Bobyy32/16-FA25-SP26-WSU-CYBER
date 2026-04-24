from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    img = ia.data.quokka(0.5)
    mul = 0.01
    augs = [
        ("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
        ("iaa.AdditiveGaussianNoise(p=1*mul)", iaa.AdditiveGaussianNoise(p=1*mul)),
        ("iaa.AdditiveGaussianNoise(p=2*mul)", iaa.AdditiveGaussianNoise(p=2*mul)),
        ("iaa.GaussianBlur(p=1*mul)", iaa.GaussianBlur(p=1*mul)),
        ("iaa.GaussianBlur(p=2*mul)", iaa.GaussianBlur(p=2*mul)),
        ("iaa.Lambda(iaa.AdditiveGaussianNoise(p=0.2*mul), 'and', iaa.AdditiveGaussianNoise(p=0.4*mul), p=1)",
         iaa.Lambda(iaa.AdditiveGaussianNoise(p=0.2*mul), "and", iaa.AdditiveGaussianNoise(p=0.4*mul), p=1)),
    ]
    print(img)


if __name__ == "__main__":
    main()