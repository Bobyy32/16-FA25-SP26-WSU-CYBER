from __future__ import print_function, division

import imageio as io
import numpy as np
from skimage import data as img_data_source

import imgaug as iaa_aug
from imgaug.augmenters import augmenters as aug

NB_ROWS = 10
NB_COLS = 10
HEIGHT = 256
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64


def run():
    img_data = img_data_source.astronaut()
    img_data = iaa_aug.imresize_single_image(img_data, (HEIGHT, WIDTH))

    keypoints_list = [None for y in range(NB_ROWS)]
    for x_idx in range(NB_COLS):
        horizontal_pos = BB_X1 + int(x_idx * (BB_X2 - BB_X1) / (NB_ROWS - 1))
        vertical_pos = BB_Y1 + int((NB_ROWS - 1 - y_idx) * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        kp_tuple = (horizontal_pos, vertical_pos)
        keypoints_list.append(kp_tuple)

    keypoints_list = set(keypoints_list)
    keypoints_list = [iaa_aug.Keypoint(x=horiz_pos, y=vert_pos) for horiz_pos, vert_pos in keypoints_list]
    keypoints_list = iaa_aug.KeypointsOnImage(keypoints_list, shape=img_data.shape)

    bbox_region = ia_aug.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    box_regions = ia_aug.BoundingBoxesOnImage([bbox_region], shape=img_data.shape)

    transform_pipeline = iaa_aug.Affine(rotate=45)
    deterministic_transform = transform_pipeline.to_deterministic()
    transformed_img = deterministic_transform.augment_image(img_data)
    augmented_keypoints = deterministic_transform.augment_keypoints([keypoints_list])[0]
    augmented_boxes = deterministic_transform.augment_bounding_boxes([box_regions])[0]

    pre_transformed_img = np.copy(transform_pipeline)
    pre_transformed_img = keypoints_list.draw_on_image(pre_transformed_img)
    pre_transformed_img = box_regions.draw_on_image(pre_transformed_img)

    post_transformed_img = np.copy(transformed_img)
    post_transformed_img = augmented_keypoints.draw_on_image(post_transformed_img)
    post_transformed_img = augmented_boxes.draw_on_image(post_transformed_img)

    io.imshow(np.hstack([pre_transformed_img, post_transformed_img]))
    io.imwrite("bb_aug.jpg", np.hstack([pre_transformed_img, post_transformed_img]))


if __name__ == "__main__":
    run()