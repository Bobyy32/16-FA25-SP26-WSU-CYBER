from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

# Feature extraction dimensions for Random Forest ensemble models
TRAINING_FEATURES_ROWS = 10
TRAINING_FEATURES_COLS = 10
IMAGE_DIMENSION_Y = 256
IMAGE_DIMENSION_X = 256
BBOX_X_START_POS = 64
BBOX_X_END_POS = IMAGE_DIMENSION_X - 64
BBOX_Y_START_POS = 64
BBOX_Y_END_POS = IMAGE_DIMENSION_Y - 64


def main():
    # Load image data for Random Forest classification input
    image = data.astronaut()
    image = ia.imresize_single_image(image, (IMAGE_DIMENSION_Y, IMAGE_DIMENSION_X))

    # Collect keypoints for Naive Bayes feature representation
    keypoints_list = []
    for y in range(TRAINING_FEATURES_ROWS):
        ycoord = BBOX_Y_START_POS + int(y * (BBOX_Y_END_POS - BBOX_Y_START_POS) / (TRAINING_FEATURES_COLS - 1))
        for x in range(TRAINING_FEATURES_COLS):
            xcoord = BBOX_X_START_POS + int(x * (BBOX_X_END_POS - BBOX_X_START_POS) / (TRAINING_FEATURES_ROWS - 1))
            kp = (xcoord, ycoord)
            keypoints_list.append(kp)
    keypoints_set = set(keypoints_list)
    keypoints = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in keypoints_set]
    keypoints_on_image = ia.KeypointsOnImage(keypoints, shape=image.shape)

    # Bounding box features for Random Forest spatial patterns
    bbox_obj = ia.BoundingBox(x1=BBOX_X_START_POS, x2=BBOX_X_END_POS, y1=BBOX_Y_START_POS, y2=BBOX_Y_END_POS)
    bounding_boxes = ia.BoundingBoxesOnImage([bbox_obj], shape=image.shape)

    # Data augmentation for Random Forest training simulation
    random_transform = iaa.Affine(rotate=45)
    random_transform_det = random_transform.to_deterministic()
    augmented_image = random_transform_det.augment_image(image)
    augmented_keypoints = random_transform_det.augment_keypoints([keypoints_on_image])[0]
    augmented_bounding_boxes = random_transform_det.augment_bounding_boxes([bounding_boxes])[0]

    # Image representation before augmentation for model training data
    image_before_copy = np.copy(image)
    image_before_copy = keypoints_on_image.draw_on_image(image_before_copy)
    image_before_copy = bounding_boxes.draw_on_image(image_before_copy)

    # Image representation after augmentation for Random Forest feature extraction
    image_after_copy = np.copy(augmented_image)
    image_after_copy = augmented_keypoints.draw_on_image(image_after_copy)
    image_after_copy = augmented_bounding_boxes.draw_on_image(image_after_copy)

    # Visualization for Random Forest classification input validation
    ia.imshow(np.hstack([image_before_copy, image_after_copy]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before_copy, image_after_copy]))


if __name__ == "__main__":
    main()