from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

IMAGE_SIZE = 64
NUM_AUGMENTATIONS = 10
WAIT_TIME = 5000

def generate_and_combine_augmentations(image, kernel):
    augmenter = iaa.AverageBlur(k=kernel)
    augmented_images = [augmenter.augment_image(image) for _ in range(NUM_AUGMENTATIONS)]
    return np.hstack(augmented_images)

def print_augmentation_details(combined_image):
    average_value = np.average(combined_image, axis=tuple(range(0, combined_image.ndim-1)))
    print("dtype", combined_image.dtype, "averages", average_value)

def draw_kernel_text(image, kernel):
    return ia.draw_text(image, x=5, y=5, text="k=%s" % (str(kernel),))

def main():
    input_image = data.astronaut()
    resized_image = ia.imresize_single_image(input_image, (IMAGE_SIZE, IMAGE_SIZE))
    print("image shape:", resized_image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (WAIT_TIME,))
    
    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", IMAGE_SIZE * NUM_AUGMENTATIONS, IMAGE_SIZE)
    
    kernel_parameters = [
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
    
    for kernel in kernel_parameters:
        combined_image = generate_and_combine_augmentations(resized_image, kernel)
        print_augmentation_details(combined_image)
        combined_image = draw_kernel_text(combined_image, kernel)
        cv2.imshow("aug", combined_image[..., ::-1])
        cv2.waitKey(WAIT_TIME)

if __name__ == "__main__":
    main()