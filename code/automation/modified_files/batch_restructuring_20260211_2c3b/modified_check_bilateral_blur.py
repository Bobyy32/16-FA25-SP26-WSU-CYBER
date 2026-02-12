from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data
import cv2

WAIT_TIME_MS = 5000
NUM_AUGMENTED_IMAGES = 10

def create_augmentation(diameter, color_stddev, spatial_stddev):
    return iaa.BilateralBlur(d=diameter, sigma_color=color_stddev, sigma_space=spatial_stddev)

def generate_augmented_images(source_image, augmentation, num_samples):
    return np.hstack([augmentation.augment_image(source_image) for _ in range(num_samples)])

def add_title_to_image(augmented_set, diameter, color_stddev, spatial_stddev):
    title = "d=%s, sc=%s, ss=%s" % (str(diameter), str(color_stddev), str(spatial_stddev))
    return ia.draw_text(augmented_set, x=5, y=5, text=title)

def main():
    source_image = data.astronaut()
    source_image = ia.imresize_single_image(source_image, (128, 128))
    print("image shape:", source_image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (WAIT_TIME_MS,))

    AUGMENTATION_CONFIGS = [
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

    window_name = "augmentation_window"
    window_width = 128 * NUM_AUGMENTED_IMAGES
    window_height = 128
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    for params in AUGMENTATION_CONFIGS:
        diameter, color_stddev, spatial_stddev = params
        augmentation = create_augmentation(diameter, color_stddev, spatial_stddev)
        augmented_set = generate_augmented_images(source_image, augmentation, NUM_AUGMENTED_IMAGES)
        print("dtype", augmented_set.dtype, "averages", np.average(augmented_set, axis=tuple(range(0, augmented_set.ndim-1))))
        augmented_set = add_title_to_image(augmented_set, diameter, color_stddev, spatial_stddev)
        cv2.imshow(window_name, augmented_set[..., ::-1])
        cv2.waitKey(WAIT_TIME_MS)

if __name__ == "__main__":
    main()