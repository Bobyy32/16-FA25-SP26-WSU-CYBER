from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


def init_image_processing():
    image = data.astronaut()
    HEIGHT = 256
    WIDTH = 256
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))
    return image


def prepare_dimensions():
    BB_X1 = 64
    BB_X2 = 192
    BB_Y1 = 64
    BB_Y2 = 192
    return BB_X1, BB_X2, BB_Y1, BB_Y2


def generate_coordinate_pairs(NB_ROWS, NB_COLS, BB_X1, BB_X2, BB_Y1, BB_Y2):
    kps = []
    for y in range(NB_ROWS):
        ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp = (xcoord, ycoord)
            kps.append(kp)
    return set(kps)


def create_keypoint_objects(coord_set):
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in coord_set]
    return kps


def initialize_keypoint_objects(kps):
    return ia.KeypointsOnImage(kps, shape=(256, 256, 3))


def create_bounding_box_object(BB_X1, BB_X2, BB_Y1, BB_Y2):
    return ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)


def prepare_augmentation_sequence():
    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    return seq_det


def apply_augmentation_transform(image, kps, bbs, seq_det):
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    return image_aug, kps_aug, bbs_aug


def draw_on_images(image_before, image_after, kp_before, kp_after, bb_before, bb_after):
    image_before = kp_before.draw_on_image(image_before)
    image_before = bb_before.draw_on_image(image_before)
    image_after = kp_after.draw_on_image(image_after)
    image_after = bb_after.draw_on_image(image_after)
    return image_before, image_after


def save_and_display_images(image_before, image_after):
    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))


def main():
    BB_X1, BB_X2, BB_Y1, BB_Y2 = prepare_dimensions()
    NB_ROWS = 10
    NB_COLS = 10
    image = init_image_processing()
    coord_set = generate_coordinate_pairs(NB_ROWS, NB_COLS, BB_X1, BB_X2, BB_Y1, BB_Y2)
    kps = create_keypoint_objects(coord_set)
    kps = initialize_keypoint_objects(kps)
    bb = create_bounding_box_object(BB_X1, BB_X2, BB_Y1, BB_Y2)
    seq_det = prepare_augmentation_sequence()
    image_before, image_after = draw_on_images(image, None, kps, None, bb, None)
    image_aug = prepare_augmentation_transform(image, kps, bb, seq_det)
    image_before, image_after = draw_on_images(np.copy(image), image_aug, kps, None, bb, None)
    save_and_display_images(image_before, image_after)


if __name__ == "__main__":
    main()