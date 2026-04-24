from __future__ import print_function, division, absolute_import

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


def display_randaugment_n_m_variations(base_image, n_values, m_values, augmentation_batch_size, display_grid_cols, random_seed):
    """
    Applies RandAugment with varying N and M parameters to a base image,
    and displays the augmented results for each N value.

    Args:
        base_image (np.array): The original image to augment.
        n_values (list[int]): A list of N parameters to iterate through.
        m_values (list[int]): A list of M parameters to iterate through for each N.
        augmentation_batch_size (int): The number of images to generate per RandAugment call.
        display_grid_cols (int): The number of columns to use when drawing the image grid.
        random_seed (int): The random state for the RandAugment augmenter.
    """
    for n_param in n_values:
        print(f"N={n_param}")
        all_augmented_images_for_n = []
        for m_param in m_values:
            # Create augmenter with specified N and M
            current_augmenter = iaa.RandAugment(n=n_param, m=m_param, random_state=random_seed)
            # Apply augmenter to a batch of the base image
            augmented_batch = current_augmenter(images=[base_image] * augmentation_batch_size)
            all_augmented_images_for_n.extend(augmented_batch)
        # Display all images generated for the current N in a grid
        ia.imshow(ia.draw_grid(all_augmented_images_for_n, cols=display_grid_cols))


def display_randaugment_m_batches(base_image, m_values, num_augmentation_batches, augmentation_batch_size, display_grid_cols, display_grid_rows, random_seed):
    """
    Applies RandAugment with varying M parameter to a base image in multiple batches,
    and displays the augmented results for each M value.

    Args:
        base_image (np.array): The original image to augment.
        m_values (list[int]): A list of M parameters to iterate through.
        num_augmentation_batches (int): The number of times to apply the augmenter for batches.
        augmentation_batch_size (int): The number of images to generate per RandAugment call.
        display_grid_cols (int): The number of columns for the visualization grid.
        display_grid_rows (int): The number of rows for the visualization grid.
        random_seed (int): The random state for the RandAugment augmenter.
    """
    for m_param in m_values:
        print(f"M={m_param}")
        current_augmenter = iaa.RandAugment(m=m_param, random_state=random_seed)
        all_augmented_images_for_m = []
        for _ in np.arange(num_augmentation_batches):
            # Apply augmenter to a batch of the base image
            augmented_batch = current_augmenter(images=[base_image] * augmentation_batch_size)
            all_augmented_images_for_m.extend(augmented_batch)
        # Display all images generated for the current M in a grid
        ia.imshow(ia.draw_grid(all_augmented_images_for_m, cols=display_grid_cols, rows=display_grid_rows))


def main():
    # Load a base image for augmentation
    base_image = ia.data.quokka(0.25)
    # Define a consistent random seed for reproducibility
    initial_random_seed = 1

    # --- First demonstration block: Vary N and M parameters ---
    # For each N value, we generate augmentations across all M values
    # and display them together in a single grid.
    print("--- Demonstrating RandAugment with varying N and M (fixed N per grid) ---")
    display_randaugment_n_m_variations(
        base_image=base_image,
        n_values=[1, 2],
        m_values=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        augmentation_batch_size=10,  # Number of images to augment per RandAugment call
        display_grid_cols=10,        # Number of columns for the visualization grid
        random_seed=initial_random_seed
    )

    # --- Second demonstration block: Vary M parameter with multiple batches ---
    # For each M value, we generate multiple batches of augmentations
    # and display them together in a single grid.
    print("\n--- Demonstrating RandAugment with varying M (fixed M per grid, multiple batches) ---")
    display_randaugment_m_batches(
        base_image=base_image,
        m_values=[0, 1, 2, 4, 8, 10],
        num_augmentation_batches=6,  # Number of times to run the augmenter for batches
        augmentation_batch_size=16,  # Number of images to augment per RandAugment call
        display_grid_cols=16,        # Number of columns for the visualization grid
        display_grid_rows=6,         # Number of rows for the visualization grid
        random_seed=initial_random_seed
    )


if __name__ == "__main__":
    main()