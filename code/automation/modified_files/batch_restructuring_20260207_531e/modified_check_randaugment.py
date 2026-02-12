from __future__ import print_function, division, absolute_import

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


def _generate_augmented_images(augmenter, images_list, num_augmentations):
    """Helper function to generate augmented images."""
    augmented_images = []
    for _ in np.arange(num_augmentations):
        augmented_images.extend(augmenter(images=images_list))
    return augmented_images


def _display_grid_images(images, cols, rows=None):
    """Helper function to display grid of images."""
    ia.imshow(ia.draw_grid(images, cols=cols, rows=rows))


def main():
    image = ia.data.quokka(0.25)

    # First loop with varying N values
    for n_val in [1, 2]:
        print("N=%d" % (n_val,))

        aug = iaa.RandAugment(n=n_val, random_state=1)
        images_aug = _generate_augmented_images(
            aug, [image] * 10, 10
        )
        _display_grid_images(images_aug, cols=10)

    # Second loop with varying M values
    for m_val in [0, 1, 2, 4, 8, 10]:
        print("M=%d" % (m_val,))
        aug = iaa.RandAugment(m=m_val, random_state=1)

        images_aug = _generate_augmented_images(
            aug, [image] * 16, 6
        )
        _display_grid_images(images_aug, cols=16, rows=6)


if __name__ == "__main__":
    main()