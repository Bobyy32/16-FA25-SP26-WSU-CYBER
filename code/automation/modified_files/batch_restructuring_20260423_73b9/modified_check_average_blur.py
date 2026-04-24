from __future__ import print_function, division

import cv2
import numpy as np
from skimage import data
import imgaug as ia
from imgaug import augmenters as iaa


class AugmentationRunner:
    """
    Refactored class to manage image augmentation processing and visualization.
    Encapsulates the core logic for applying augmentations and displaying results.
    """

    def __init__(self, time_per_step=5000, nb_augs_per_image=10):
        self.time_per_step = time_per_step
        self.nb_augs_per_image = nb_augs_per_image

    def load_and_preprocess_image(self):
        """Load image, resize, and prepare for augmentation processing."""
        image = data.astronaut()
        image = ia.imresize_single_image(image, (64, 64))
        print("image shape:", image.shape)
        print("Press any key or wait %d ms to proceed to the next image." % self.time_per_step)

    def configure_window(self, window_name="aug", width=64 * self.nb_augs_per_image, height=64):
        """Configure and initialize OpenCV display window."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    def apply_augmentation(self, base_image, k_value):
        """Apply average blur augmentation for the specified k parameter."""
        aug = iaa.AverageBlur(k=k_value)
        augmented_images = [aug.augment_image(base_image) for _ in range(self.nb_augs_per_image)]
        img_aug = np.hstack(augmented_images)
        
        print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim - 1))))
        
        title = "k=%s" % str(k_value)
        img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)
        
        return img_aug[..., ::-1]  # RGB to BGR conversion

    def display_and_wait(self, image):
        """Display augmented image and wait for specified duration."""
        cv2.imshow("aug", image)
        cv2.waitKey(self.time_per_step)


def main():
    """Main execution function for the augmentation system."""
    runner = AugmentationRunner()
    runner.load_and_preprocess_image()

    # Define augmentation parameters as structured data
    k_parameters = [
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

    # Configure display window before processing
    runner.configure_window()

    # Process each augmentation parameter
    for k in k_parameters:
        processed_image = runner.apply_augmentation(image, k)
        runner.display_and_wait(processed_image)


if __name__ == "__main__":
    main()