from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data
import cv2

TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10


def _create_augmenters():
    """Creates a list of BilateralBlur augmenters with varying parameters."""
    configs = [
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
    augmenters = []
    for d_param, sigma_color_param, sigma_space_param in configs:
        augmenter = iaa.BilateralBlur(
            d=d_param,
            sigma_color=sigma_color_param,
            sigma_space=sigma_space_param
        )
        augmenters.append((augmenter, d_param, sigma_color_param, sigma_space_param))
    return augmenters


def _display_image(image, title, window_name="aug"):
    """Displays an image with a given title in a specified window."""
    image_with_text = ia.draw_text(image, x=5, y=5, text=title)
    cv2.imshow(window_name, image_with_text[..., ::-1])  # Convert RGB to BGR
    cv2.waitKey(TIME_PER_STEP)


def main():
    initial_image = data.astronaut()
    initial_image = ia.imresize_single_image(initial_image, (128, 128))
    print("Initial image shape:", initial_image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))

    augmenters_with_params = _create_augmenters()

    window_name = "aug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 128 * NB_AUGS_PER_IMAGE, 128)

    for augmenter, d_param, sigma_color_param, sigma_space_param in augmenters_with_params:
        augmented_images = [augmenter.augment_image(initial_image) for _ in range(NB_AUGS_PER_IMAGE)]
        combined_image = np.hstack(augmented_images)

        # Calculate average for each channel
        avg_values = np.average(combined_image, axis=tuple(range(0, combined_image.ndim - 1)))
        print(f"dtype: {combined_image.dtype}, averages: {avg_values}")

        title = f"d={d_param}, sc={sigma_color_param}, ss={sigma_space_param}"
        _display_image(combined_image, title, window_name)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()