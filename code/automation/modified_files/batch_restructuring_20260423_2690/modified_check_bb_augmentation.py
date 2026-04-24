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


def get_image():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))
    return image


def create_keypoint_grid():
    kps = []
    for y in range(NB_ROWS):
        ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp = (xcoord, ycoord)
            kps.append(kp)
    kps = set(kps)
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    return kps


def build_keypoint_objects(kps):
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    return kps


def build_bounding_boxes():
    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)
    return bbs


def create_image_transform_seq():
    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    return seq_det


def transform_image(image):
    image_aug = seq_det.augment_image(image)
    return image_aug


def transform_keypoints(kps):
    kps_aug = seq_det.augment_keypoints([kps])[0]
    return kps_aug


def transform_bounding_boxes(bbs):
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    return bbs_aug


def draw_keypoints_and_boxes(image, kps, bbs):
    image_before = np.copy(image)
    image_before = kps.draw_on_image(image_before)
    image_before = bbs.draw_on_image(image_before)
    return image_before


def draw_augmented_image(image, kps_aug, bbs_aug):
    image_after = np.copy(image)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)
    return image_after


def display_image(image1, image2):
    ia.imshow(np.hstack([image1, image2]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image1, image2]))


if __name__ == "__main__":
    image = get_image()
    kps = create_keypoint_grid()
    kps = build_keypoint_objects(kps)
    bbs = build_bounding_boxes()
    seq_det = create_image_transform_seq()
    image = transform_image(image)
    kps_aug = transform_keypoints(kps)
    bbs_aug = transform_bounding_boxes(bbs)
    image_before = draw_keypoints_and_boxes(image, kps, bbs)
    image_after = draw_augmented_image(image, kps_aug, bbs_aug)
    display_image(image_before, image_after)