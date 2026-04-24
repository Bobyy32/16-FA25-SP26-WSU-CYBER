from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

WIDTH = 256
HEIGHT = 256
NB_COLS = 10
NB_ROWS = 10
BB_X1 = 64
BB_Y1 = 64
BB_X2 = WIDTH - 64
BB_Y2 = HEIGHT - 64


def _generate_keypoints(grid_cols, grid_rows, x_start, x_end, y_start, y_end):
    kps = []
    for y in range(grid_cols):
        ycoord = int(y_start + y * (y_end - y_start) / (grid_cols - 1))
        for x in range(grid_rows):
            xcoord = int(x_start + x * (x_end - x_start) / (grid_rows - 1))
            kps.append((xcoord, ycoord))
    return set(kps), [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]


def _prepare_keypoints(kps, image_shape):
    return ia.KeypointsOnImage(kps, shape=image_shape)


def _prepare_bounding_boxes(bb, image_shape):
    return ia.BoundingBoxesOnImage([bb], shape=image_shape)


def _apply_augmentation(seq_det, image, keypoints, bboxes):
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([keypoints])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bboxes])[0]
    return image_aug, kps_aug, bbs_aug


def _draw_on_image(image, keypoints, bboxes):
    image = keypoints.draw_on_image(image)
    image = bboxes.draw_on_image(image)
    return image


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps_set, kps = _generate_keypoints(NB_COLS, NB_ROWS, BB_X1, BB_X2, BB_Y1, BB_Y2)
    kps_obj = _prepare_keypoints(kps, image.shape)

    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs_obj = _prepare_bounding_boxes(bb, image.shape)

    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    image_aug, kps_aug, bbs_aug = _apply_augmentation(seq_det, image, kps_obj, bbs_obj)

    image_before = np.copy(image)
    image_before = _draw_on_image(image_before, kps_obj, bbs_obj)

    image_after = np.copy(image_aug)
    image_after = _draw_on_image(image_after, kps_aug, bbs_aug)

    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))
    ia.imshow(np.hstack([image_before, image_after]))


if __name__ == "__main__":
    main()