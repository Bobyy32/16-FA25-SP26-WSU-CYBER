from __future__ import print_function, division

# Required imports for image processing and analysis capabilities
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    """
    The main function loads an image and iterates through augmentation configurations.
    It demonstrates the application of impulse noise with a quokka image.
    """
    # Load a random quokka image with 0.5 noise level
    img = ia.data.quokka(0.5)
    
    # Initialize the multiplier variable for adjusting noise probability
    mul = 0.01
    
    # List of augmentation objects and their string representations
    augs = [
        ("iaa.ImpulseNoise(p=0*mul)", iaa.ImpulseNoise(p=0*mul)),
        ("iaa.ImpulseNoise(p=1*mul)", iaa.ImpulseNoise(p=1*mul)),
        ("iaa.ImpulseNoise(p=2*mul)", iaa.ImpulseNoise(p=2*mul)),
        ("iaa.ImpulseNoise(p=3*mul)", iaa.ImpulseNoise(p=3*mul)),
        ("iaa.ImpulseNoise(p=(0*mul, 1*mul))", iaa.ImpulseNoise(p=(0*mul, 1*mul))),
        ("iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul])", iaa.ImpulseNoise(p=[0*mul, 1*mul, 2*mul]))
    ]

    # Apply the specified augmentation to multiple image copies
    for descr, aug in augs:
        # Print the current configuration description
        print(descr)
        imgs_aug = aug.augment_images([img] * 16)
        # Display the output grid for visual evaluation
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    # Execute the main workflow when this script is run directly
    main()