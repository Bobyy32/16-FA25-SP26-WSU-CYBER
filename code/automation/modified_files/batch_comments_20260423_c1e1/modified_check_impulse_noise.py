from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    """
    Load a sample image and demonstrate various impulse noise augmentation techniques.
    This function displays the effect of different noise probability configurations
    to visualize how image augmentation impacts visual results.
    """
    # Initialize a sample quokka image with moderate brightness
    img = ia.data.quokka(0.5)
    
    # Define a scaling multiplier to adjust probability parameters
    mul = 0.01
    
    # Configure multiple noise augmentation strategies with their corresponding instances
    augs = [
        ("ImpulseNoise using zero probability", iaa.ImpulseNoise(p=0*mul)),
        ("ImpulseNoise using one probability", iaa.ImpulseNoise(p=1*mul)),
        ("ImpulseNoise with double probability", iaa.ImpulseNoise(p=2*mul)),
        ("ImpulseNoise with triple probability", iaa.ImpulseNoise(p=3*mul)),
        ("ImpulseNoise with tuple probability distribution", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
        ("ImpulseNoise with list probability distribution", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
    ]
    
    # Process each augmentation strategy by printing its identifier
    for descr, aug in augs:
        print(descr)
        
        # Apply augmentation to multiple image copies using the configured strategy
        imgs_aug = aug.augment_images([img] * 16)
        
        # Display the augmented images in a grid layout for visual inspection
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()