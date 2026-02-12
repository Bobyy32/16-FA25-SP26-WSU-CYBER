from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def _generate_variations_with_single_param(base_image, augmenter_class, param_name, param_values):
    """
    Generates a list of augmented images by varying a single parameter of an augmenter.

    Args:
        base_image (np.ndarray): The input image to augment.
        augmenter_class (type): The imgaug augmenter class (e.g., iaa.MultiplyHueAndSaturation).
        param_name (str): The name of the parameter to vary (e.g., 'mul', 'mul_hue').
        param_values (np.ndarray): An array of values for the specified parameter.

    Returns:
        list: A list of augmented images.
    """
    processed_images = []
    for value in param_values:
        # Create augmenter instance with dynamic keyword argument
        augmenter_instance = augmenter_class(**{param_name: value})
        augmented_image = augmenter_instance.augment_image(base_image)
        processed_images.append(augmented_image)
    return processed_images


def _append_augmented_batch(result_list, base_image, augmenter_instance, num_copies):
    """
    Augments a batch of identical images using a given augmenter and extends a list.

    Args:
        result_list (list): The list to extend with augmented images.
        base_image (np.ndarray): The input image to copy and augment.
        augmenter_instance (iaa.Augmenter): An imgaug augmenter instance.
        num_copies (int): The number of copies of the base_image to augment.
    """
    batch_images = [base_image] * num_copies
    augmented_batch = augmenter_instance.augment_images(batch_images)
    result_list.extend(augmented_batch)


def _display_grid_of_images(image_list, rows):
    """
    Displays a grid of images using imgaug's utility functions.

    Args:
        image_list (list): A list of images (np.ndarray) to display.
        rows (int): The number of rows for the image grid.
    """
    grid_image = ia.draw_grid(image_list, rows=rows)
    ia.imshow(grid_image)


def main():
    original_image = ia.quokka_square((128, 128))
    collected_augmented_images = []

    # --- Part 1: Varying MultiplyHueAndSaturation parameters ---

    # Varying 'mul' parameter
    param_values_mul = np.linspace(0.0, 2.0, 10)
    collected_augmented_images.extend(
        _generate_variations_with_single_param(
            original_image, iaa.MultiplyHueAndSaturation, 'mul', param_values_mul
        )
    )

    # Varying 'mul_hue' parameter
    param_values_mul_hue = np.linspace(0.0, 5.0, 10)
    collected_augmented_images.extend(
        _generate_variations_with_single_param(
            original_image, iaa.MultiplyHueAndSaturation, 'mul_hue', param_values_mul_hue
        )
    )

    # Varying 'mul_saturation' parameter
    param_values_mul_saturation = np.linspace(0.0, 5.0, 10)
    collected_augmented_images.extend(
        _generate_variations_with_single_param(
            original_image, iaa.MultiplyHueAndSaturation, 'mul_saturation', param_values_mul_saturation
        )
    )

    _display_grid_of_images(collected_augmented_images, rows=3)

    # --- Part 2: Applying MultiplyHue and MultiplySaturation directly ---

    collected_augmented_images = []  # Reset for the next display

    num_augmentations_per_type = 10

    _append_augmented_batch(
        collected_augmented_images,
        original_image,
        iaa.MultiplyHue(),
        num_augmentations_per_type
    )

    _append_augmented_batch(
        collected_augmented_images,
        original_image,
        iaa.MultiplySaturation(),
        num_augmentations_per_type
    )

    _display_grid_of_images(collected_augmented_images, rows=2)


if __name__ == "__main__":
    main()