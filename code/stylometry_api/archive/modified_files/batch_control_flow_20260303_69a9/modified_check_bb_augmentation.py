from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as aug

NB_ROWS = 10
NB_COLS = 10
HEIGHT = 256
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64


def execute():
    img = data.astronaut()
    img = ia.imresize_single_image(img, (HEIGHT, WIDTH))

    keypoints = []
    for y in range(NB_ROWS):
        ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp = (xcoord, ycoord)
            keypoints.append(kp)
    keypoints = set(keypoints)
    keypoints = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in keypoints]
    keypoints = ia.KeypointsOnImage(keypoints, shape=img.shape)

    box = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    boxes = ia.BoundingBoxesOnImage([box], shape=img.shape)

    seq = aug.Affine(rotate=45)
    seq_det = seq.to_deterministic()
    img_aug = seq_det.augment_image(img)
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
    boxes_aug = seq_det.augment_bounding_boxes([boxes])[0]

    img_before = np.copy(img)
    img_before = keypoints.draw_on_image(img_before)
    img_before = boxes.draw_on_image(img_before)

    img_after = np.copy(img_aug)
    img_after = keypoints_aug.draw_on_image(img_after)
    img_after = boxes_aug.draw_on_image(img_after)

    ia.imshow(np.hstack([img_before, img_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([img_before, img_after]))


if __name__ == "__main__":
    execute()