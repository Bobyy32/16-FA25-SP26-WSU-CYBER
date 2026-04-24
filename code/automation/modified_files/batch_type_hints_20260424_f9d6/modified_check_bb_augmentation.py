from __future__ import print_function, division

import typing
from typing import Optional, Union, List, Set, Tuple, Any

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


def main(image_input: Optional[np.ndarray] = None) -> Any:
    if image_input is None:
        image = data.astronaut()
        image = ia.imresize_single_image(image, (HEIGHT, WIDTH))
    else:
        image = image_input
    
    kps: Optional[ia.KeypointsOnImage] = None
    
    for y in range(NB_ROWS):
        ycoord: int = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord: int = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp: Tuple[int, int] = (xcoord, ycoord)
            kps = ia.KeypointsOnImage(kps)
            kps = List[kp]
            kps.append(kp)
    
    kps = ia.KeypointsOnImage(kps, shape=image.shape)

    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    seq: ia.Augmenter = iaa.Affine(rotate=45)
    seq_det: ia.Deterministic = seq.to_deterministic()
    image_aug: np.ndarray = seq_det.augment_image(image)
    kps_aug: ia.KeypointsOnImage = seq_det.augment_keypoints([kps])[0]
    bbs_aug: ia.BoundingBoxesOnImage = seq_det.augment_bounding_boxes([bbs])[0]

    image_before: np.ndarray = np.copy(image)
    image_before = kps.draw_on_image(image_before)
    image_before = bbs.draw_on_image(image_before)

    image_after: np.ndarray = np.copy(image_aug)
    image_after = kps_aug.draw_on_image(image_after)
    image_after = bbs_aug.draw_on_image(image_after)

    ia.imshow(np.hstack([image_before, image_after]))
    imageio.imwrite("bb_aug.jpg", np.hstack([image_before, image_after]))


if __name__ == "__main__":
    main()