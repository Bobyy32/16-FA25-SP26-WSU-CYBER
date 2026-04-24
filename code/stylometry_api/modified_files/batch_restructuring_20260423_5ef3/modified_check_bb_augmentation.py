from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

NB_ROWS = 10
NB_COLS = 10
HEIGHT = 256
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64


def create_keypoints_on_image(image, bb_x1, bb_x2, bb_y1, bb_y2, nb_rows, nb_cols):
    kps = []
    for y in range(nb_rows):
        ycoord = bb_y1 + int(y * (bb_y2 - bb_y1) / (nb_cols - 1))
        for x in range(nb_cols):
            xcoord = bb_x1 + int(x * (bb_x2 - bb_x1) / (nb_rows - 1))
            kp = (xcoord, ycoord)
            kps.append(kp)
    kps = set(kps)
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    kps = ia.KeypointsOnImage(kps, shape=image.shape)
    return kps


def create_boxes_on_image(image, bb_x1, bb_x2, bb_y1, bb_y2):
    bb = ia.BoundingBox(x1=bb_x1, x2=bb_x2, y1=bb_y1, y2=bb_y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)
    return bbs


def apply_augmentation_and_draw(image, kps, bbs, seq_det):
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    image_before = np.copy(image)
    image_before = kps.draw_on_image(image_before)
    image_before = bbs.draw_on_image(image_before)

    image_after = np.copy(image_aug)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)

    return image_before, image_after


if __name__ == "__main__":
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps = create_keypoints_on_image(image, BB_X1, BB_X2, BB_Y1, BB_Y2, NB_ROWS, NB_COLS)
    bbs = create_boxes_on_image(image, BB_X1, BB_X2, BB_Y1, BB_Y2)

    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()

    image_before, image_after = apply_augmentation_and_draw(image, kps, bbs, seq_det)

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))