from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    # Grab some sample quokka image from the package data
    img = ia.data.quokka(0.5)
    
    # A scaling factor for parameter adjustment
    mul = 0.01
    
    # Set up list of augmentation configurations - each with descriptive label and actual instance
    augs = [
        ("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
        ("iaa.ImpulseNoise(p=2*mul)", iaa.ImpulseNoise(p=2*mul)),
        ("iaa.ImpulseNoise(p=3*mul)", iaa.ImpulseNoise(p=3*mul)),
        ("iaa.ImpulseNoise(p=(0*mul, 1*mul))", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
        ("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
    ]
    
    # Loop through each augmentation setup and process results
    for descr, aug in augs:
        print(descr)
        # Apply augmentation to 16 copies of the original image
        imgs_aug = aug.augment_images([img] * 16)
        # Display grid of augmented results
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()