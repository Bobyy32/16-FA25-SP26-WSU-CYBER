from __future__ import print_function, division

import time
from itertools import cycle

import cv2
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

# --- Constants ---
SEGMENTS_PER_STEP = 1
TIME_PER_STEP = 10

# --- Helper Functions ---

def _load_source_image():
    """Loads the initial image for processing.
    Returns:
        np.ndarray: The loaded image in BGR format.
    """
    # data.astronaut() returns RGB, convert to BGR for OpenCV
    source_image = data.astronaut()[..., ::-1]
    return source_image

def _initialize_display_window(window_name, initial_image, delay_ms):
    """Initializes an OpenCV window and displays the initial image.

    Args:
        window_name (str): The name of the OpenCV window.
        initial_image (np.ndarray): The image to display initially.
        delay_ms (int): Time in milliseconds to wait for a key event.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, initial_image)
    cv2.waitKey(delay_ms)

def _create_superpixel_augmenter(num_segments_value):
    """Creates a Superpixels augmenter with a given number of segments.

    Args:
        num_segments_value (int): The number of segments for Superpixels.
    Returns:
        iaa.Superpixels: An instance of the Superpixels augmenter.
    """
    return iaa.Superpixels(p_replace=0.75, n_segments=num_segments_value)

def _apply_and_display_augmentation(source_image, augmenter_instance, display_param_value, window_name, delay_ms):
    """Applies an augmenter to an image, measures time, adds text, and displays it.

    Args:
        source_image (np.ndarray): The original image to augment.
        augmenter_instance (imgaug.augmenters.Augmenter): The augmenter to apply.
        display_param_value (any): The parameter value to display on the image and print.
        window_name (str): The name of the OpenCV window for display.
        delay_ms (int): Time in milliseconds to wait for a key event.
    """
    time_start = time.time()
    augmented_image = augmenter_instance.augment_image(source_image)
    elapsed_time = time.time() - time_start
    print("Augmented with param %s in %.4fs" % (str(display_param_value), elapsed_time))

    # Add the current parameter value as text to the augmented image
    augmented_image = ia.draw_text(augmented_image, x=5, y=5, text="%s" % (str(display_param_value),))

    cv2.imshow(window_name, augmented_image)
    cv2.waitKey(delay_ms)

# --- Main Logic ---

def main():
    # 1. Load the initial data
    source_image = _load_source_image()
    print(source_image.shape)

    # 2. Setup the display window
    display_window_name = "aug"
    _initialize_display_window(display_window_name, source_image, TIME_PER_STEP)

    # 3. Define the parameter range for augmentation (generic terms)
    param_start = 1
    param_end = 200
    param_step = SEGMENTS_PER_STEP

    # 4. Loop through parameters, apply augmentation, and display results
    for current_param_value in cycle(reversed(np.arange(param_start, param_end, param_step))):
        augmenter_instance = _create_superpixel_augmenter(current_param_value)
        _apply_and_display_augmentation(
            source_image,
            augmenter_instance,
            current_param_value,
            display_window_name,
            TIME_PER_STEP
        )


if __name__ == "__main__":
    main()