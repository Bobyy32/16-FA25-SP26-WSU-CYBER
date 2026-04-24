from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


"""
This script demonstrates the application of various impulse noise augmentation
techniques on test images using the imgaug library. It generates and displays
a grid of augmented image results for different impulse noise probability configurations.
"""


def main():
    """
    Load a test image and apply various impulse noise augmentations with 
    increasing probability parameters. Each augmentation is applied to 16 
    copies of the input image, and the results are displayed in a grid 
    format for visual comparison.
    """
    # Load and initialize a quokka image as the test data
    img = ia.data.quokka(0.5)
    
    # Set the base multiplier factor to scale probability parameters
    mul = 0.01
    
    # Define list of augmentation configurations combining string descriptions
    # with their corresponding actual implementation objects
    augs = [
        ("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
        ("iaa.ImpulseNoise(p=2*mul)", iaa.ImpulseNoise(p=2*mul)),
        ("iaa.ImpulseNoise(p=3*mul)", iaa.ImpulseNoise(p=3*mul)),
        ("iaa.ImpulseNoise(p=(0*mul, 1*mul))", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
        ("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
    ]
    
    # Iterate through each augmentation configuration and execute transformations
    for descr, aug in augs:
        # Print the current augmentation description for reference
        print(descr)
        
        # Apply the impulse noise augmentation to 16 copies of the original image
        imgs_aug = aug.augment_images([img] * 16)
        
        # Display the augmented images in a grid format
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()