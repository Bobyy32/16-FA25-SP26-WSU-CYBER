from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

HEIGHT = 256
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64
NB_ROWS = 10
NB_COLS = 10


def generate_keypoints(image, nb_rows, nb_cols, bb_x1, bb_x2, bb_y1, bb_y2):
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


def create_bounding_boxes(bb_x1, bb_x2, bb_y1, bb_y2):
    bb = ia.BoundingBox(x1=bb_x1, x2=bb_x2, y1=bb_y1, y2=bb_y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)
    return bbs


def apply_augmentation(seq):
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    return image_aug, kps_aug, bbs_aug


def process_and_draw(image, kps, bbs):
    image_before = np.copy(image)
    image_before = kps.draw_on_image(image_before)
    image_before = bbs.draw_on_image(image_before)
    return image_before


def process_and_draw_augmented(image_aug, kps_aug, bbs_aug):
    image_after = np.copy(image_aug)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)
    return image_after


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps = generate_keypoints(image, NB_ROWS, NB_COLS, BB_X1, BB_X2, BB_Y1, BB_Y2)
    bbs = create_bounding_boxes(BB_X1, BB_X2, BB_Y1, BB_Y2)

    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    image_aug, kps_aug, bbs_aug = apply_augmentation(seq_det)
    image_aug = image_aug
    kps_aug = kps_aug
    bbs_aug = bbs_aug

    image_before = process_and_draw(image, kps, bbs)
    image_after = process_and_draw_augmented(image_aug, kps_aug, bbs_aug)
    image_after = np.hstack([image_before, image_after])

    ia.imshow(image_after)
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))


if __name__ == "__main__":
    main()