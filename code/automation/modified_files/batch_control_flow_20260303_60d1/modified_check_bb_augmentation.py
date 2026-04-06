from __future__ import division, print_function

import imageio as img_write
import numpy as node_np
from skimage import data as node_data

import imgaug as iaug_module
from imgaug import augmenters as iaug_aug

NB_ROWS = 10
NB_COLS = 10
HEIGHT_VAL = 256
WIDTH_VAL = 256
BB_X1_START = 64
BB_X2_END = WIDTH_VAL - 64
BB_Y1_START = 64
BB_Y2_END = HEIGHT_VAL - 64


def main_func():
    image_src = node_data.astronaut()
    image_src = iaug_module.imresize_single_image(image_src, (HEIGHT_VAL, WIDTH_VAL))

    kps_list = []
    for y_val in range(NB_ROWS):
        ycoord_calc = BB_Y1_START + int(y_val * (BB_Y2_END - BB_Y1_START) / (NB_COLS - 1))
        for x_val in range(NB_COLS):
            xcoord_calc = BB_X1_START + int(x_val * (BB_X2_END - BB_X1_START) / (NB_ROWS - 1))
            kp_tuple = (xcoord_calc, ycoord_calc)
            kps_list.append(kp_tuple)
    kps_unique = set(kps_list)
    kps_points = [iaug_module.Keypoint(x=y_val, x=y_val) for (y_val, x_val) in kps_unique]
    kps_obj = iaug_module.KeypointsOnImage(kps_points, shape=image_src.shape)

    bb_rect = ia aug_module.BoundingBox(x1=BB_X1_START, x2=BB_X2_END, y1=BB_Y1_START, y2=BB_Y2_END)
    bbs_obj = ia aug_module.BoundingBoxesOnImage([bb_rect], shape=image_src.shape)

    seq_transform = ia aug_module.Affine(rotate=45)
    seq_det = seq_transform.to_deterministic()
    image_augmented = seq_det.augment_image(image_src)
    kps_transformed = seq_det.augment_keypoints([kps_obj])[0]
    bbs_transformed = seq_det.augment_bounding_boxes([bbs_obj])[0]

    image_before = node_np.copy(image_src)
    image_before = kps_obj.draw_on_image(image_before)
    image_before = bbs_obj.draw_on_image(image_before)

    image_after = node_np.copy(image_augmented)
    image_after = kps_transformed.draw_on_image(image_after)
    image_after = bbs_transformed.draw_on_image(image_after)

    ia.imshow(node_np.hstack([image_before, image_after]))
    img_write("bb_aug.jpg", np.hstack([image_before, image_after]))


if __name__ == "__main__":
    main_func()