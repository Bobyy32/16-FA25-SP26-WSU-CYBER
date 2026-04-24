from __future__ import print_function, division

import numpy as np
import imageio
from skimage import data
import imgaug as ia
from imgaug import augmenters as iaa


def generate_grid_keypoints(nb_rows, nb_cols, height, width, bb_x1, bb_x2, bb_y1, bb_y2):
    """Generate keypoint coordinates on a regular grid within bounding box"""
    keypoint_list = []
    for row_idx in range(nb_rows):
        y_coord = bb_y1 + int(row_idx * (bb_y2 - bb_y1) / (nb_rows - 1))
        for col_idx in range(nb_cols):
            x_coord = bb_x1 + int(col_idx * (bb_x2 - bb_x1) / (nb_cols - 1))
            keypoint_list.append((x_coord, y_coord))
    return keypoint_list


def convert_to_keypoint_objects(keypoint_coords):
    """Convert coordinate tuples to imgaug Keypoint objects"""
    keypoint_objects = [ia.Keypoint(x=x, y=y) for x, y in keypoint_coords]
    return ia.KeypointsOnImage(keypoint_objects)


def prepare_bounding_boxes(bb):
    """Create bounding boxes object for image augmentation"""
    bboxes = [bb]
    return ia.BoundingBoxesOnImage(bboxes)


def augment_transformations(image, keypoints, bounding_boxes, transform_sequence):
    """Apply deterministic transformation to all image components"""
    augmented_image = transform_sequence.augment_image(image)
    transformed_keypoints = transform_sequence.augment_keypoints([keypoints])[0]
    transformed_bounding_boxes = transform_sequence.augment_bounding_boxes([bounding_boxes])[0]
    return augmented_image, transformed_keypoints, transformed_bounding_boxes


def display_and_save_comparison(image_before, keypoints_before, bounding_boxes_before,
                                 image_after, keypoints_after, bounding_boxes_after):
    """Render side-by-side comparison and save to file"""
    combined = np.hstack([image_before, image_after])
    ia.imshow(combined)
    imageio.imwrite("bb_aug.jpg", combined)


def create_keypoints_and_boxes(NB_ROWS, NB_COLS, HEIGHT, WIDTH, BB_X1, BB_X2, BB_Y1, BB_Y2):
    """Factory function to generate all keypoints and bounding boxes"""
    grid_coords = generate_grid_keypoints(NB_ROWS, NB_COLS, HEIGHT, WIDTH, 
                                           BB_X1, BB_X2, BB_Y1, BB_Y2)
    keypoints = convert_to_keypoint_objects(grid_coords)
    bounding_box = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bounding_boxes = prepare_bounding_boxes(bounding_box)
    return keypoints, bounding_boxes


def main():
    """Main execution function"""
    # Configuration parameters
    NB_ROWS = 10
    NB_COLS = 10
    HEIGHT = 256
    WIDTH = 256
    BB_X1 = 64
    BB_X2 = WIDTH - 64
    BB_Y1 = 64
    BB_Y2 = HEIGHT - 64
    
    # Load and resize input image
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))
    
    # Prepare keypoints and bounding boxes
    keypoints, bounding_boxes = create_keypoints_and_boxes(
        NB_ROWS, NB_COLS, HEIGHT, WIDTH, BB_X1, BB_X2, BB_Y1, BB_Y2
    )
    
    # Setup augmentation sequence
    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    
    # Apply augmentation to all components
    image_aug, kps_aug, bbs_aug = augment_transformations(
        image, keypoints, bounding_boxes, seq_det
    )
    
    # Generate before/after comparisons
    image_before = np.copy(image)
    image_before = keypoints.draw_on_image(image_before)
    image_before = bounding_boxes.draw_on_image(image_before)
    
    image_after = np.copy(image_aug)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)
    
    # Display and save results
    display_and_save_comparison(
        image_before, keypoints, bounding_boxes,
        image_after, kps_aug, bbs_aug
    )


if __name__ == "__main__":
    main()