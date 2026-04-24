from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

TOTAL_ROWS = 10
TOTAL_COLS = 10
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BBOX_START_X1 = 64
BBOX_END_X2 = IMAGE_WIDTH - 64
BBOX_START_Y1 = 64
BBOX_END_Y2 = IMAGE_HEIGHT - 64


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    keypoints_list = []
    for row_idx in range(TOTAL_ROWS):
        row_coord = BBOX_START_Y1 + int(row_idx * (BBOX_END_Y2 - BBOX_START_Y1) / (TOTAL_COLS - 1))
        for col_idx in range(TOTAL_COLS):
            col_coord = BBOX_START_X1 + int(col_idx * (BBOX_END_X2 - BBOX_START_X1) / (TOTAL_ROWS - 1))
            kp = (col_coord, row_coord)
            keypoints_list.append(kp)
    keypoints_list = set(keypoints_list)
    keypoints_list = [ia.Keypoint(x=col_coord, y=row_coord) for (col_coord, row_coord) in keypoints_list]
    keypoints_list = ia.KeypointsOnImage(keypoints_list, shape=image.shape)

    bbox = ia.BoundingBox(x1=BBOX_START_X1, x2=BBOX_END_X2, y1=BBOX_START_Y1, y2=BBOX_END_Y2)
    bboxes = ia.BoundingBoxesOnImage([bbox], shape=image.shape)

    affine_transform = iaa.Affine(rotate=45)
    affine_transform_det = affine_transform.to_deterministic()
    image_aug = affine_transform_det.augment_image(image)
    keypoints_aug = affine_transform_det.augment_keypoints([keypoints_list])[0]
    bboxes_aug = affine_transform_det.augment_bounding_boxes([bboxes])[0]

    image_before = np.copy(image)
    image_before = keypoints_list.draw_on_image(image_before)
    image_before = bboxes.draw_on_image(image_before)

    image_after = np.copy(image_aug)
    image_after = keypoints_aug.draw_on_image(image_after)
    image_after = bboxes_aug.draw_on_image(image_after)

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))


if __name__ == "__main__":
    main()