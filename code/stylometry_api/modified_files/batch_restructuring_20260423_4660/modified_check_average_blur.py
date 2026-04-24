from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

# Define constant values for configuration
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10

# Helper function to create and prepare the base image
def prepare_base_image():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return image

# Helper function to display processing message with time
def show_proceed_message():
    print("Press any key or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))

# Helper function to generate multiple augmented images for a given kernel
def apply_augmentation(image, ki, nb_aug_count):
    augmented_list = [iaa.AverageBlur(k=ki).augment_image(image) for _ in range(nb_aug_count)]
    return np.hstack(augmented_list)

# Helper function to compute mean value statistics across image dimensions
def compute_image_statistics(img_data):
    return np.average(img_data, axis=tuple(range(0, img_data.ndim-1)))

# Helper function to format and embed text onto the augmented image
def label_image(img_data, kernel):
    title = "k=%s" % (str(kernel),)
    img_data = ia.draw_text(img_data, x=5, y=5, text=title)
    return img_data

# Helper function to display the final augmented image with correct color format
def display_image(img_data):
    img_data = img_data[..., ::-1]  # Convert BGR to RGB for proper display
    cv2.imshow("aug", img_data)
    return

# Main processing routine
def execute_visualization_pipeline():
    image = prepare_base_image()
    print("image shape:", image.shape)
    show_proceed_message()

    # Initialize OpenCV window
    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", IMAGE_WIDTH*NB_AUGS_PER_IMAGE, IMAGE_HEIGHT)

    # Define augmentation kernels
    k = [
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

    for ki in k:
        img_aug = apply_augmentation(image, ki, NB_AUGS_PER_IMAGE)
        stats = compute_image_statistics(img_aug)
        print("dtype", img_aug.dtype, "averages", stats)

        img_aug = label_image(img_aug, ki)
        display_image(img_aug)
        cv2.waitKey(TIME_PER_STEP)


# Entry point execution
if __name__ == "__main__":
    execute_visualization_pipeline()