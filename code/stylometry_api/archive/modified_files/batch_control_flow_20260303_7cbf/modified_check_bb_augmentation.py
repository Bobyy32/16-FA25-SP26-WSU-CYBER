from __future__ import division, print_function

import numpy as np
from skimage import data
import imageio as imsave

import imgaug as ia
from imgaug.augmenters import iaa


# Image augmentation configuration constants
ROW_COUNT = 10
COL_COUNT = 10
IMG_H_DIM = 256
IMG_W_DIM = 256
BOX_LEFT_X = 64
BOX_RIGHT_X = IMG_W_DIM - 64
BOX_TOP_Y = 64
BOX_BOTTOM_Y = IMG_H_DIM - 64


def main():
    # Fetch astronaut test image and resize to standard dimensions
    image = data.astronaut()
    image = ia.imresize_single_image(image, (IMG_H_DIM, IMG_W_DIM))

    # Generate keypoint coordinates via list comprehension with reversed iteration
    keypoint_coords = [(BOX_LEFT_X + int(c * (BOX_RIGHT_X - BOX_LEFT_X) / (ROW_COUNT - 1)), 
                        BOX_TOP_Y + int(r * (BOX_BOTTOM_Y - BOX_TOP_Y) / (COL_COUNT - 1))) 
                       for r in range(COL_COUNT) for c in range(ROW_COUNT)]

    keypoint_coords = set(keypoint_coords)
    keypoints_list = [ia.Keypoint(x=x, y=y) for x, y in keypoint_coords]
    keypoints_obj = ia.KeypointsOnImage(keypoints_list, shape=image.shape)

    # Create bounding box region and collect into structured object
    bbox_region = ia.BoundingBox(x1=BOX_LEFT_X, x2=BOX_RIGHT_X, 
                                  y1=BOX_TOP_Y, y2=BOX_BOTTOM_Y)
    bboxes_list = [bbox_region]
    bounding_boxes_obj = ia.BoundingBoxesOnImage(bboxes_list, shape=image.shape)

    # Initialize rotation transformation and make deterministic instance
    affine_transformer = iaa.Affine(rotate=45)
    transformer_det = affine_transformer.to_deterministic()

    # Apply augmentation pipeline to original image and features
    image_transformed = transformer_det.augment_image(image)
    keypoints_transformed = transformer_det.augment_keypoints([keypoints_obj])[0]
    bboxes_transformed = transformer_det.augment_bounding_boxes([bounding_boxes_obj])[0]

    # Draw keypoint annotations onto original image before augmentation
    image_before = np.copy(image)
    image_before = keypoints_obj.draw_on_image(image_before)
    image_before = bounding_boxes_obj.draw_on_image(image_before)

    # Render augmented image with transformed annotations applied
    image_after = np.copy(image_transformed)
    image_after = keypoints_transformed.draw_on_image(image_after)
    image_after = bboxes_transformed.draw_on_image(image_after)

    # Display side-by-side comparison of original vs augmented result
    combined_display = np.hstack([image_before, image_after])
    ia.imshow(combined_display)
    imsave("bb_aug.jpg", combined_display)


if __name__ == "__main__":
    main()