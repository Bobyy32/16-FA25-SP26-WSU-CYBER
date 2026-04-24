from __future__ import print_function, division

"""
Import modules for image augmentation library functionality
"""

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    """
    Main function to demonstrate image augmentation capabilities
    using impulse noise with varying probability parameters
    """
    img = ia.data.quokka(0.5)
    mul = 0.01
    
    # Create array of augmentation configurations with descriptions
    augs = [
        ("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
        ("iaa.ImpulseNoise(p=2*mul)", iaa.ImpulseNoise(p=2*mul)),
        ("iaa.ImpulseNoise(p=3*mul)", iaa.ImpulseNoise(p=3*mul)),
        ("iaa.ImpulseNoise(p=(0*mul, 1*mul))", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
        ("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
    ]
    
    for descr, aug in augs:
        # Output the description string for each augmentation variant
        print(descr)
        # Apply augmentation to sixteen identical image copies
        imgs_aug = aug.augment_images([img] * 16)
        # Display grid of augmented images
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()