from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

NB_ROWS: int = 10
NB_COLS: int = 10
HEIGHT: int = 256
WIDTH: int = 256
BB_X1: int = 64
BB_X2: int = WIDTH - 64
BB_Y1: int = 64
BB_Y2: int = HEIGHT - 64


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps: list[tuple[int, int]] = []
    for y in range(NB_ROWS):
        ycoord: int = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord: int = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp: tuple[int, int] = (xcoord, ycoord)
            kps.append(kp)
    kps: set[tuple[int, int]] = set(kps)
    kps: list[ia.Keypoint] = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    kps: ia.KeypointsOnImage = ia.KeypointsOnImage(kps, shape=image.shape)

    bb: ia.BoundingBox = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs: ia.BoundingBoxesOnImage = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    seq: iaa.Affine = iaa.Affine(rotate=45)
    seq_det: iaa.Affine = seq.to_deterministic()
    image_aug = seq_det.augment_image(image)
    kps_aug = seq_det.augment_keypoints([kps])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    image_before = np.copy(image)
    image_before = kps.draw_on_image(image_before)
    image_before = bbs.draw_on_image(image_before)

    image_after = np.copy(image_aug)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))


if __name__ == "__main__":
    main()