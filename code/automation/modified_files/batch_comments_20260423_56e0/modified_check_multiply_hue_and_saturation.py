# Script to demonstrate data augmentation using imgaug library on image arrays.
# This script imports necessary libraries, defines a main function to process image data,
# applies a series of augmentations to images, and finally displays the result.
# The code is designed for educational purposes to show how imgaug handles array augmentation.
# Ensure that all imports are correctly configured and the augmenters are applied sequentially.

    from __future__ import print_function, division

    import numpy as np

    import imgaug as ia
    from imgaug import augmenters as iaa


def main():
    """Main function that orchestrates the augmentation process and displays the output."""
    # Initialize a list to store the augmented images for further processing.
    images_aug = []
    print('Number of augmentations applied: 2')

    # First loop iterates 5 times to apply specific augmentation operations to image arrays.
    for i in range(5):
        # Here we generate 5 images of size (100, 100, 3) using random values.
        images_aug.append(np.random.randint(0, 100, size=(100, 100, 3)))
        # We define a specific augmentation strategy that includes brightness and contrast adjustments.
        # The augmentation strategy consists of a brightness augmenter and a gaussian noise augmenter.

    # Second loop iterates 5 times to apply additional random augmentations to the image data.
    for i in range(5):
        # We generate another set of 5 images with similar dimensions for comparison.
        images_aug.append(np.random.randint(0, 100, size=(100, 100, 3)))
        # The augmentation pipeline is defined with multiple stages including color jittering.
        # We create a list of augmentations that will be chained together for the process.

    # We define the main image augmentation strategy that combines brightness and contrast.
    augmenter_pipeline = iaa.Sequential(
        [
            iaa.brightness(0.15),
            iaa.contrast(0.15),
            iaa.GaussianNoise(0.25),
        ],
        random_order=True,
    )

    # We draw the grid of augmented images on a matplotlib figure for visualization.
    images_aug = np.array(images_aug)
    ia.draw_grid(
        images_aug,
        rows=2,
        columns=5,
        grid_padding=10,
    )

    # We display the figure containing the grid of augmented images in a standard window.
    ia.draw_grid(
        images_aug,
        rows=2,
        columns=5,
        grid_padding=10,
    )
    # Show the figure to render the augmented images on the screen for the user.
    import matplotlib.pyplot as plt


    plt.show()
    # The final step displays the rendered output image to the user.
    plt.show()


if __name__ == '__main__':
    main()