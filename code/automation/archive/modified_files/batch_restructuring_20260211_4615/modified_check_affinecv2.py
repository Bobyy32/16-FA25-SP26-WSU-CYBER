from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
from skimage import data
import cv2

# --- Configuration Constants ---
# These constants define the setup for the image, keypoint grid, and bounding box.
GRID_ROWS = 10
GRID_COLS = 10
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 256
BOUND_X1 = 64
BOUND_X2 = IMAGE_WIDTH - 64
BOUND_Y1 = 64
BOUND_Y2 = IMAGE_HEIGHT - 64

# --- Helper Functions ---

def load_and_resize_base_image(target_height, target_width):
    """
    Loads a base image (astronaut) from skimage.data and resizes it
    to the specified target dimensions.
    """
    base_image = data.astronaut()
    resized_image = ia.imresize_single_image(base_image, (target_height, target_width))
    return resized_image

def generate_keypoints_grid(num_rows, num_cols, x1_bound, y1_bound, x2_bound, y2_bound, image_shape):
    """
    Generates a grid of keypoints within specified bounding box coordinates.
    The keypoints are distributed uniformly across the bounds.
    Returns an `ia.KeypointsOnImage` object.
    """
    keypoint_coords = []

    # Calculate step sizes for even distribution within the bounds
    # Handle single row/column case to prevent division by zero
    y_step = (y2_bound - y1_bound) / (num_rows - 1) if num_rows > 1 else 0
    x_step = (x2_bound - x1_bound) / (num_cols - 1) if num_cols > 1 else 0

    for y_idx in range(num_rows):
        y_coord = y1_bound + int(y_idx * y_step)
        for x_idx in range(num_cols):
            x_coord = x1_bound + int(x_idx * x_step)
            keypoint_coords.append((x_coord, y_coord))

    # Convert raw coordinates to imgaug Keypoint objects
    ia_keypoints = [ia.Keypoint(x=x_coord, y=y_coord) for (x_coord, y_coord) in keypoint_coords]
    return ia.KeypointsOnImage(ia_keypoints, shape=image_shape)

def create_single_bounding_box_on_image(x1_bound, y1_bound, x2_bound, y2_bound, image_shape):
    """
    Creates a single bounding box from given coordinates.
    Returns an `ia.BoundingBoxesOnImage` object containing this bounding box.
    """
    bounding_box = ia.BoundingBox(x1=x1_bound, x2=x2_bound, y1=y1_bound, y2=y2_bound)
    return ia.BoundingBoxesOnImage([bounding_box], shape=image_shape)

def get_predefined_affine_augmenters():
    """
    Returns a list of predefined `iaa.AffineCv2` augmenters
    demonstrating various transformation parameters.
    """
    augmenter_list = [
        iaa.AffineCv2(rotate=45),
        iaa.AffineCv2(translate_px=20),
        iaa.AffineCv2(translate_percent=0.1),
        iaa.AffineCv2(scale=1.2),
        iaa.AffineCv2(scale=0.8),
        iaa.AffineCv2(shear=45),
        iaa.AffineCv2(rotate=45, cval=256),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_CONSTANT),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_REPLICATE),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_REFLECT),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_REFLECT_101),
        iaa.AffineCv2(translate_px=20, mode=cv2.BORDER_WRAP),
        iaa.AffineCv2(translate_px=20, mode="constant"),
        iaa.AffineCv2(translate_px=20, mode="replicate"),
        iaa.AffineCv2(translate_px=20, mode="reflect"),
        iaa.AffineCv2(translate_px=20, mode="reflect_101"),
        iaa.AffineCv2(translate_px=20, mode="wrap"),
        iaa.AffineCv2(scale=0.5, order=cv2.INTER_NEAREST),
        iaa.AffineCv2(scale=0.5, order=cv2.INTER_LINEAR),
        iaa.AffineCv2(scale=0.5, order=cv2.INTER_CUBIC),
        iaa.AffineCv2(scale=0.5, order=cv2.INTER_LANCZOS4),
        iaa.AffineCv2(scale=0.5, order="nearest"),
        iaa.AffineCv2(scale=0.5, order="linear"),
        iaa.AffineCv2(scale=0.5, order="cubic"),
        iaa.AffineCv2(scale=0.5, order="lanczos4"),
        iaa.AffineCv2(rotate=45, translate_px=20, scale=1.2),
        iaa.AffineCv2(rotate=45, translate_px=20, scale=0.8),
        iaa.AffineCv2(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineCv2(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineCv2(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineCv2(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL)
    ]
    return augmenter_list

def apply_and_visualize_augmentation(augmenter_instance, original_image, original_keypoints, original_bboxes):
    """
    Applies a single augmentation sequence to the original image, keypoints, and bounding boxes.
    Generates two images: one showing the original data and one showing the augmented data,
    both with drawn keypoints and bounding boxes.
    Returns these two images stacked horizontally.
    """
    # Make the augmenter deterministic for consistent application across image and annotations
    deterministic_augmenter = augmenter_instance.to_deterministic()

    # Augment the image, keypoints, and bounding boxes
    augmented_image = deterministic_augmenter.augment_image(original_image)
    augmented_keypoints = deterministic_augmenter.augment_keypoints([original_keypoints])[0]
    augmented_bboxes = deterministic_augmenter.augment_bounding_boxes([original_bboxes])[0]

    # Create the 'before' visualization: original image with original annotations
    image_before_viz = np.copy(original_image)
    image_before_viz = original_keypoints.draw_on_image(image_before_viz)
    image_before_viz = original_bboxes.draw_on_image(image_before_viz)

    # Create the 'after' visualization: augmented image with augmented annotations
    image_after_viz = np.copy(augmented_image)
    image_after_viz = augmented_keypoints.draw_on_image(image_after_viz)
    image_after_viz = augmented_bboxes.draw_on_image(image_after_viz)

    # Stack 'before' and 'after' images horizontally for comparison
    return np.hstack((image_before_viz, image_after_viz))

def main():
    """
    Main function to orchestrate the image augmentation and visualization process.
    """
    # 1. Load and prepare the base image
    base_image = load_and_resize_base_image(IMAGE_HEIGHT, IMAGE_WIDTH)

    # 2. Generate initial keypoints grid
    initial_keypoints = generate_keypoints_grid(
        GRID_ROWS, GRID_COLS, BOUND_X1, BOUND_Y1, BOUND_X2, BOUND_Y2, base_image.shape
    )

    # 3. Create initial bounding box
    initial_bounding_boxes = create_single_bounding_box_on_image(
        BOUND_X1, BOUND_Y1, BOUND_X2, BOUND_Y2, base_image.shape
    )

    # 4. Get the list of augmenters to apply
    augmentation_sequences = get_predefined_affine_augmenters()

    # 5. Process each augmentation sequence and collect results
    all_visualization_pairs = []
    for current_augmenter in augmentation_sequences:
        combined_viz_image = apply_and_visualize_augmentation(
            current_augmenter, base_image, initial_keypoints, initial_bounding_boxes
        )
        all_visualization_pairs.append(combined_viz_image)

    # 6. Stack all individual visualization pairs vertically for a final overview
    final_combined_visualization = np.vstack(all_visualization_pairs)

    # 7. Display and save the final visualization
    ia.imshow(final_combined_visualization)
    imageio.imwrite("affinecv2_transformed_output.jpg", final_combined_visualization)

if __name__ == "__main__":
    main()