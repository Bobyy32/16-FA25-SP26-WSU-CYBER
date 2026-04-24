from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa


class ImageAugmentationDemo:
    """Handles image augmentation display with configurable parameters."""
    
    TIME_PER_STEP = 5000
    NB_AUGS_PER_IMAGE = 10
    
    def __init__(self, time_per_step: int, nb_augs_per_image: int):
        """Initialize the demo with configuration parameters."""
        self.time_per_step = time_per_step
        self.nb_augs_per_image = nb_augs_per_image
    
    def get_augmentation_configurations(self) -> list:
        """Returns the list of AverageBlur configurations to apply."""
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
        return k
    
    def load_and_preprocess_image(self) -> np.ndarray:
        """Loads and preprocesses the base image."""
        try:
            image = data.astronaut()
            image = ia.imresize_single_image(image, (64, 64))
            print("Image shape:", image.shape)
            return image
        except Exception as e:
            print(f"Failed to load/preprocess image: {e}")
            raise
    
    def display_configuration_parameters(self, image: np.ndarray, title: str) -> np.ndarray:
        """Displays the augmented image with configuration label."""
        try:
            img_aug = ia.draw_text(image, x=5, y=5, text=title)
            return img_aug
        except Exception as e:
            print(f"Failed to draw text: {e}")
            raise
    
    def process_augmentations(self, image: np.ndarray, k_config: int) -> np.ndarray:
        """Applies average blur augmentation to the image."""
        try:
            aug = iaa.AverageBlur(k=k_config)
            img_aug = [aug.augment_image(image) for _ in range(self.nb_augs_per_image)]
            img_aug = np.hstack(img_aug)
            print(f"dtype: {img_aug.dtype}")
            avg_val = np.average(img_aug, axis=tuple(range(0, img_aug.ndim - 1)))
            print(f"averages: {avg_val}")
            return img_aug
        except Exception as e:
            print(f"Failed to process augmentations: {e}")
            raise
    
    def display_augmented_image(self, augmented_image: np.ndarray, title: str) -> None:
        """Displays the augmented image with proper conversion."""
        try:
            cv2.imshow("aug", augmented_image[..., ::-1])
            cv2.waitKey(self.time_per_step)
        except Exception as e:
            print(f"Failed to display augmented image: {e}")
            raise
    
    def setup_opencv(self) -> None:
        """Sets up OpenCV window and dimensions."""
        try:
            cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("aug", 64 * self.nb_augs_per_image, 64)
        except Exception as e:
            print(f"Failed to setup OpenCV window: {e}")
            raise


def main():
    """Main entry point for the image augmentation demonstration."""
    try:
        demo = ImageAugmentationDemo(TIME_PER_STEP, NB_AUGS_PER_IMAGE)
        image = demo.load_and_preprocess_image()
        print(f"Press any key or wait {demo.time_per_step} ms to proceed.")

        cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("aug", 64 * demo.nb_augs_per_image, 64)

        k_config_list = demo.get_augmentation_configurations()

        for ki in k_config_list:
            try:
                img_aug = demo.process_augmentations(image, ki)
                title = f"k={ki}"
                img_aug = demo.display_configuration_parameters(img_aug, title)
                demo.display_augmented_image(img_aug, title)
            except Exception:
                continue

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in main execution: {e}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()