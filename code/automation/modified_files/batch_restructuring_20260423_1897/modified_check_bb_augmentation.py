from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

def setup_grid_coords():
    BB_X1 = 64
    BB_X2 = 256 - 64
    BB_Y1 = 64
    BB_Y2 = 256 - 64
    NB_ROWS = 10
    NB_COLS = 10
    HEIGHT = 256
    WIDTH = 256
    return BB_X1, BB_X2, BB_Y1, BB_Y2, NB_ROWS, NB_COLS, HEIGHT, WIDTH

def construct_kps(bboxes, coords):
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in coords]
    kps = ia.KeypointsOnImage(kps, shape=bboxes.shape)
    return kps

def generate_kps(bboxes, coords):
    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=bboxes.shape)
    kps = []
    for y in range(NB_ROWS):
        ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp = (xcoord, ycoord)
            kps.append(kp)
    kps = set(kps)
    return kps, bbs

def apply_transform(image, kps):
    seq = iaa.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    return image_aug, kps_aug

def draw_kps_on_image(image, kps):
    image = kps.draw_on_image(image)
    return image

def draw_bboxes_on_image(image, bbs):
    image = bbs.draw_on_image(image)
    return image

def main():
    BB_X1, BB_X2, BB_Y1, BB_Y2, NB_ROWS, NB_COLS, HEIGHT, WIDTH = setup_grid_coords()

    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))
    kps, bbs = generate_kps(image, image.shape)

    kps = construct_kps(image, kps)

    image_before = np.copy(image)
    image_before = draw_kps_on_image(image_before, kps)
    image_before = draw_bboxes_on_image(image_before, bbs)

    image_after, kps_aug = apply_transform(image, kps)
    image_after = np.copy(image_aug)
    image_after = draw_kps_on_image(image_after, kps_aug)
    image_after = draw_bboxes_on_image(image_after, bbs_aug)

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))

if __name__ == "__main__":
    main()