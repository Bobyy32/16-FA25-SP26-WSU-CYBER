from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data
import cv2

TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10
DISPLAY_ITEM_WIDTH = 128
DISPLAY_ITEM_HEIGHT = 128


def load_input_data(target_size):
    """Loads an image (astronaut) and resizes it to the target_size."""
    item = data.astronaut()
    item = ia.imresize_single_image(item, target_size)
    print("Input item shape:", item.shape)
    return item


def get_augmentation_configurations():
    """Returns a list of parameter tuples for BilateralBlur augmentation."""
    configurations = [
        (1, 75, 75),
        (3, 75, 75),
        (5, 75, 75),
        (10, 75, 75),
        (10, 25, 25),
        (10, 250, 150),
        (15, 75, 75),
        (15, 150, 150),
        (15, 250, 150),
        (20, 75, 75),
        (40, 150, 150),
        ((1, 5), 75, 75),
        (5, (10, 250), 75),
        (5, 75, (10, 250)),
        (5, (10, 250), (10, 250)),
        (10, (10, 250), (10, 250)),
    ]
    return configurations


def initialize_display_window(window_name, item_width, item_height, num_items_to_display):
    """
    Initializes and resizes an OpenCV display window.
    Prints instructions for user interaction.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, item_width * num_items_to_display, item_height)
    print("Press any key or wait %d ms to proceed to the next display." % (TIME_PER_STEP,))


def process_and_display_augmentations(
    input_item,
    d_param,
    sigma_color_param,
    sigma_space_param,
    num_augmentations,
    window_name,
    display_wait_time
):
    """
    Applies a BilateralBlur augmentation with given parameters to an input item
    multiple times, stacks the results, adds a title, and displays them.
    """
    current_augmenter = iaa.BilateralBlur(d=d_param, sigma_color=sigma_color_param, sigma_space=sigma_space_param)

    augmented_items = [current_augmenter.augment_image(input_item) for _ in range(num_augmentations)]
    display_content = np.hstack(augmented_items)

    print("dtype", display_content.dtype, "averages", np.average(display_content, axis=tuple(range(0, display_content.ndim - 1))))

    title_text = "d=%s, sc=%s, ss=%s" % (str(d_param), str(sigma_color_param), str(sigma_space_param))
    display_content = ia.draw_text(display_content, x=5, y=5, text=title_text)

    cv2.imshow(window_name, display_content[..., ::-1])  # OpenCV expects BGR, so convert RGB to BGR
    cv2.waitKey(display_wait_time)


def main():
    target_item_size = (DISPLAY_ITEM_WIDTH, DISPLAY_ITEM_HEIGHT)
    input_item = load_input_data(target_item_size)

    augmentation_configurations = get_augmentation_configurations()

    window_name = "aug"
    initialize_display_window(window_name, DISPLAY_ITEM_WIDTH, DISPLAY_ITEM_HEIGHT, NB_AUGS_PER_IMAGE)

    for (d, sigma_color, sigma_space) in augmentation_configurations:
        process_and_display_augmentations(
            input_item,
            d,
            sigma_color,
            sigma_space,
            NB_AUGS_PER_IMAGE,
            window_name,
            TIME_PER_STEP
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()