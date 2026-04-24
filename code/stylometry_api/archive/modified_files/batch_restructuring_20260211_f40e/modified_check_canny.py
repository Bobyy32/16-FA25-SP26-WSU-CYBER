from __future__ import print_function, division
import imgaug as ia
import imgaug.augmenters as iaa


def _display_augmentation_example(description_text, augmenter_instance, input_image):
    """
    Applies an augmenter to multiple copies of an image and displays the results in a grid.

    Args:
        description_text (str): A description to print before showing the augmented images.
        augmenter_instance (iaa.Augmenter): The augmenter to apply.
        input_image (np.ndarray): The base image to augment.
    """
    print(description_text)
    # Apply the augmenter to 25 copies of the input image
    augmented_results = augmenter_instance(images=[input_image] * (5 * 5))
    ia.imshow(ia.draw_grid(augmented_results))


def main():
    # Load a common image once to be used across all examples
    base_image = ia.quokka_square((128, 128))

    # Define a common colorizer instance
    binary_colorizer = iaa.RandomColorsBinaryImageColorizer(
        color_true=255, color_false=0)

    # Example 1: Canny with black and white colorizer
    description_1 = "alpha=1.0, black and white"
    augmenter_1 = iaa.Canny(alpha=1.0, colorizer=binary_colorizer)
    _display_augmentation_example(description_1, augmenter_1, base_image)

    # Example 2: Canny with random color (default)
    description_2 = "alpha=1.0, random color"
    augmenter_2 = iaa.Canny(alpha=1.0)
    _display_augmentation_example(description_2, augmenter_2, base_image)

    # Example 3: Canny with specific Sobel kernel size range and black and white
    description_3 = "alpha=1.0, sobel ksize=[3, 13], black and white"
    # Note: original code used [3, 7] for sobel_kernel_size, description says [3, 13].
    # Sticking to the original code's parameter value.
    augmenter_3 = iaa.Canny(alpha=1.0, sobel_kernel_size=[3, 7],
                            colorizer=binary_colorizer)
    _display_augmentation_example(description_3, augmenter_3, base_image)

    # Example 4: Canny with specific single Sobel kernel size and black and white
    description_4 = "alpha=1.0, sobel ksize=3, black and white"
    augmenter_4 = iaa.Canny(alpha=1.0, sobel_kernel_size=3,
                            colorizer=binary_colorizer)
    _display_augmentation_example(description_4, augmenter_4, base_image)

    # Example 5: Canny with fully random parameters
    description_5 = "fully random"
    augmenter_5 = iaa.Canny()
    _display_augmentation_example(description_5, augmenter_5, base_image)


if __name__ == "__main__":
    main()