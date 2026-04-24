from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


# ============================================================================
# Configuration Constants
# ============================================================================
GRID_ROWS = 10
GRID_COLS = 10
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BB_START_X = 64
BB_END_X = IMAGE_WIDTH - 64
BB_START_Y = 64
BB_END_Y = IMAGE_HEIGHT - 64


# ============================================================================
# Image and Data Generation Helpers
# ============================================================================
def create_resized_image():
    """Load the astronaut image and resize to target dimensions."""
    image = data.astronaut()
    image = ia.imresize_single_image(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return image


def generate_keypoints_on_grid(image_shape):
    """Generate keypoints at evenly distributed positions within the bounding box."""
    kp_list = []
    grid_row_step = (BB_END_Y - BB_START_Y) // (GRID_ROWS - 1)
    grid_col_step = (BB_END_X - BB_START_X) // (GRID_COLS - 1)

    for y_idx in range(GRID_ROWS):
        ycoord = BB_START_Y + int(y_idx * grid_row_step)
        for x_idx in range(GRID_COLS):
            xcoord = BB_START_X + int(x_idx * grid_col_step)
            kp = (xcoord, ycoord)
            kp_list.append(kp)

    # Convert to set to eliminate duplicates, then back to Keypoint list
    unique_kps = [ia.Keypoint(x=xcoord, y=ycoord)
                  for xcoord, ycoord in set(kp_list)]
    
    return ia.KeypointsOnImage(unique_kps, shape=image_shape)


def initialize_augmenter():
    """Configure the deterministic augmentation sequence."""
    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    return seq_det


# ============================================================================
# Augmentation and Annotation Processing
# ============================================================================
def augment_image_and_annotations(image, keypoints, bounding_boxes, augmenter):
    """Apply augmentations to image, keypoints, and bounding boxes."""
    augmented_image = augmenter.augment_image(image)
    augmented_keypoints = augmenter.augment_keypoints([keypoints])[0]
    augmented_bounding_boxes = augmenter.augment_bounding_boxes([bounding_boxes])[0]
    return augmented_image, augmented_keypoints, augmented_bounding_boxes


def annotate_image(image, keypoints, bounding_boxes):
    """Draw keypoints and bounding boxes on the image."""
    annotated_image = np.copy(image)
    annotated_image = keypoints.draw_on_image(annotated_image)
    annotated_image = bounding_boxes.draw_on_image(annotated_image)
    return annotated_image


# ============================================================================
# Main Execution Pipeline
# ============================================================================
def main():
    # --- 1. Load and prepare image ---
    image = create_resized_image()

    # --- 2. Generate keypoint data ---
    kps = generate_keypoints_on_grid(image.shape)

    # --- 3. Define bounding box ---
    bb = ia.BoundingBox(
        x1=BB_START_X,
        x2=BB_END_X,
        y1=BB_START_Y,
        y2=BB_END_Y
    )
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    # --- 4. Set up deterministic augmentations ---
    seq = initialize_augmenter()

    # --- 5. Apply augmentations ---
    image_aug, kps_aug, bbs_aug = augment_image_and_annotations(
        image, kps, bbs, seq
    )

    # --- 6. Create annotated image versions ---
    image_before = annotate_image(image, kps, bbs)
    image_after = annotate_image(image_aug, kps_aug, bbs_aug)

    # --- 7. Display and save results ---
    combined_image = np.hstack([image_before, image_after])
    ia.imshow(combined_image)
    imageio.imwrite("bb_aug.jpg", combined_image)


if __name__ == "__main__":
    main()