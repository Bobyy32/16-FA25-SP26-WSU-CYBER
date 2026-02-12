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


def _create_keypoints(image_shape):
    kps = []
    for y in range(NB_ROWS):
        ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp = (xcoord, ycoord)
            kps.append(kp)
    kps = set(kps)
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    return ia.KeypointsOnImage(kps, shape=image_shape)


def _create_bounding_boxes(image_shape):
    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    return ia.BoundingBoxesOnImage([bb], shape=image_shape)


def _create_augmentation_sequence():
    return iaa.Affine(rotate=45)


def _draw_on_image(image, keypoints, bounding_boxes):
    image_copy = np.copy(image)
    image_copy = keypoints.draw_on_image(image_copy)
    image_copy = bounding_boxes.draw_on_image(image_copy)
    return image_copy


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps = _create_keypoints(image.shape)
    bbs = _create_bounding_boxes(image.shape)

    seq = _create_augmentation_sequence()
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    image_before = _draw_on_image(image, kps, bbs)
    image_after = _draw_on_image(image_aug, kps_aug, bbs_aug)

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))


if __name__ == "__main__":
    main()