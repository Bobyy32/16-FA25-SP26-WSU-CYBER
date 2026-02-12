from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

# --- Configuration Constants ---
TIME_PER_STEP = 5000  # Milliseconds to wait before showing the next image
NB_AUGS_PER_IMAGE = 10  # Number of augmented versions to show for each parameter set
DISPLAY_WINDOW_NAME = "aug_display" # Name for the OpenCV window

# --- Helper Functions ---

def prepare_source_image(image_loader_func, target_dimensions):
    """
    Loads an image using the provided loader function and resizes it.

    Args:
        image_loader_func (callable): A function that returns the source image.
        target_dimensions (tuple): A tuple (width, height) for resizing the image.

    Returns:
        numpy.ndarray: The prepared (loaded and resized) source image.
    """
    source_image = image_loader_func()
    source_image = ia.imresize_single_image(source_image, target_dimensions)
    print(f"Source image shape: {source_image.shape}")
    return source_image

def setup_display_window(window_name, width, height, wait_message_ms):
    """
    Initializes and configures an OpenCV display window.

    Args:
        window_name (str): The name of the window.
        width (int): The desired width of the window.
        height (int): The desired height of the window.
        wait_message_ms (int): Time in milliseconds for the user prompt.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    print(f"Press any key or wait {wait_message_ms} ms to proceed to the next image.")

def process_and_display_augmentation(
    base_image_data, augmentation_parameter, num_variants_to_show,
    display_window_name, display_wait_time_ms
):
    """
    Applies a specific augmentation, combines multiple variants, adds text,
    and displays the result in an OpenCV window.

    Args:
        base_image_data (numpy.ndarray): The original image to augment.
        augmentation_parameter (any): The parameter specific to the augmentation (e.g., 'k' for AverageBlur).
        num_variants_to_show (int): How many augmented versions to generate and display.
        display_window_name (str): The name of the OpenCV window to display results in.
        display_wait_time_ms (int): How long to display the image (in milliseconds) before proceeding.
    """
    # Create the augmenter instance using the given parameter
    current_augmenter = iaa.AverageBlur(k=augmentation_parameter)

    # Augment the base image multiple times
    augmented_variants = [
        current_augmenter.augment_image(base_image_data)
        for _ in range(num_variants_to_show)
    ]

    # Combine the augmented images horizontally for display
    combined_augmented_image = np.hstack(augmented_variants)

    # Optional: print debug information about the combined image
    print(
        f"dtype: {combined_augmented_image.dtype}, "
        f"averages: {np.average(combined_augmented_image, axis=tuple(range(0, combined_augmented_image.ndim-1)))}"
    )

    # Add a title/label to the combined image for context
    title_text = f"k={str(augmentation_parameter)}"
    image_with_title = ia.draw_text(combined_augmented_image, x=5, y=5, text=title_text)

    # Display the image (OpenCV expects BGR, imgaug uses RGB)
    cv2.imshow(display_window_name, image_with_title[..., ::-1])
    cv2.waitKey(display_wait_time_ms)

# --- Main Execution Flow ---

def main():
    # Define specific image loading and resizing parameters
    initial_image_source = data.astronaut
    processing_target_shape = (64, 64)

    # Prepare the initial image data
    base_processed_image = prepare_source_image(initial_image_source, processing_target_shape)

    # Define the list of augmentation parameters to test
    average_blur_k_values = [
        1, 2, 4, 8, 16,
        (8, 8), (1, 8),
        ((1, 1), (8, 8)),
        ((1, 16), (1, 16)),
        ((1, 16), 1)
    ]

    # Calculate window dimensions based on the processed image size and number of variants
    window_display_width = processing_target_shape[0] * NB_AUGS_PER_IMAGE
    window_display_height = processing_target_shape[1]

    # Set up the OpenCV display window
    setup_display_window(
        DISPLAY_WINDOW_NAME,
        window_display_width,
        window_display_height,
        TIME_PER_STEP
    )

    # Iterate through each augmentation parameter, apply it, and display results
    for param_value in average_blur_k_values:
        process_and_display_augmentation(
            base_processed_image,
            param_value,
            NB_AUGS_PER_IMAGE,
            DISPLAY_WINDOW_NAME,
            TIME_PER_STEP
        )

    # Clean up OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()