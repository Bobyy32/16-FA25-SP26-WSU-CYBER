from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

# Configuration constants
TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10

# Blurry parameter configurations
BLUR_PARAMS = [
    1,
    2,
    4,
    8,
    16,
    (8, 8),
    (1, 8),
    ((1, 1), (8, 8)),
    ((1, 16), (1, 16)),
    ((1, 16), 1)
]

def load_and_preprocess_image():
    """Load astronaut image and resize to 64x64"""
    image = data.astronaut()
    image = ia.imresize_single_image(image, (64, 64))
    return image

def display_augmented_images(image, blur_params):
    """Apply augmentations and display in CV2 window"""
    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 64 * NB_AUGS_PER_IMAGE, 64)

    for param in blur_params:
        aug = iaa.AverageBlur(k=param)
        augmented_batch = [aug.augment_image(image) for _ in range(NB_AUGS_PER_IMAGE)]
        augmented_batch = np.hstack(augmented_batch)
        avg_value = np.average(augmented_batch, axis=tuple(range(0, augmented_batch.ndim - 1)))
        print(f"dtype {augmented_batch.dtype}, averages {avg_value}")

        title = f"k={str(param)}"
        augmented_batch = ia.draw_text(augmented_batch, x=5, y=5, text=title)

        # Display BGR version (RGB for processing, converted to BGR for display)
        cv2.imshow("aug", augmented_batch[..., ::-1])
        cv2.waitKey(TIME_PER_STEP)

def main():
    """Main execution flow"""
    # Load and preprocess image
    image = load_and_preprocess_image()

    # Display augmented images
    display_augmented_images(image, BLUR_PARAMS)

if __name__ == "__main__":
    main()